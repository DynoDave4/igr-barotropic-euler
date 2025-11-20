// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "general/forall.hpp"
#include "laghos_solver.hpp"
#include "laghos_IGR_solver.hpp"
#include "linalg/kernels.hpp"
#include <unordered_map>

#ifdef MFEM_USE_MPI

// for benchmark timing purposes; for a regular run this can be a no-op
#define LAGHOS_DEVICE_SYNC MFEM_DEVICE_SYNC

namespace mfem
{
	

class MassMatrix1 : public MatrixCoefficient
{
   private:
      GridFunction &phi; // vector-valued GridFunction
	  double alpha;
   public:
      MassMatrix1(GridFunction &phi_, double alpha_) : MatrixCoefficient(phi_.FESpace()->GetVDim()), phi(phi_), alpha(alpha_) {}

   virtual void Eval(DenseMatrix &M, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
	  int dim = phi.FESpace()->GetVDim();

      DenseMatrix Jac;
      phi.GetVectorGradient(T, Jac);  // Jac(i,j) = d phi_i / dx_j
	  Jac(0,0) += 1.0; Jac(1,1) += 1.0;

	  DenseMatrix Jac_inv, Jac_invT;
	  Jac_inv = Jac;
	  Jac_inv.Invert();
	  
	  Jac_invT = Jac_inv;
	  Jac_invT.Transpose();
	  
	  M.SetSize(dim, dim);
	  M = 0.0;
	  AddMult(Jac_inv, Jac_invT, M);
	  for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         { M(i,j) *= alpha; }
	  }	 
   }
};

class LambdaDivPart : public VectorCoefficient
{
   private:
      GridFunction &phi; // vector-valued GridFunction
      GridFunction &gradX; // vector-valued GridFunction
      GridFunction &gradY; // vector-valued GridFunction
	  double alpha;
   public:
      LambdaDivPart(GridFunction &phi_, GridFunction &gradX_, GridFunction &gradY_, double alpha_) : 
           VectorCoefficient(phi_.FESpace()->GetVDim()), phi(phi_), gradX(gradX_), gradY(gradY_), alpha(alpha_) {}

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
     T.SetIntPoint(&ip);
	  int dim = phi.FESpace()->GetVDim();

     
     //Hessian and Jacobian calculations at a point
     DenseMatrix Jac, HessX, HessY;
     phi.GetVectorGradient(T, Jac);  // Jac(i,j) = d phi_i / dx_j
	 Jac(0,0) += 1.0; Jac(1,1) += 1.0;
     gradX.GetVectorGradient(T, HessX);
     gradY.GetVectorGradient(T, HessY);
	  DenseMatrix Jac_inv; Jac_inv = Jac; Jac_inv.Invert();

	  //Ax and Ay store pieces of each hessian
	  DenseMatrix Ax(dim, dim), Ay(dim, dim);
      Ax(0,0) = HessX(0,0); Ax(0,1) = HessX(0,1);
      Ay(0,0) = HessX(1,0); Ay(0,1) = HessX(1,1);
      Ax(1,0) = HessY(0,0); Ax(1,1) = HessY(0,1);
      Ay(1,0) = HessY(1,0); Ay(1,1) = HessY(1,1);
	  
	  //Multiply by the inverse Jacobian
	  DenseMatrix Bx(dim, dim), By(dim, dim);
	  Bx = 0.0; AddMult(Ax, Jac_inv, Bx);
	  By = 0.0; AddMult(Ay, Jac_inv, By);
	  
     //Multiply by the inverse Jacobian again, this is the (-) derivative of the inverse of the hessian
	  DenseMatrix Cx(dim, dim), Cy(dim, dim);
      Cx = 0.0; AddMult(Jac_inv, Bx, Cx);  // Cx = [DΦ]^{-1} ∂x[DΦ] [DΦ]^{-1}
	  Cy = 0.0; AddMult(Jac_inv, By, Cy);  // Cy = [DΦ]^{-1} ∂y[DΦ] [DΦ]^{-1}
	  
     //Output
	  V.SetSize(dim);    //Coefficients of Lambda2
	  V(0) = alpha*(-1*Cx(0,0) - Cy(1,0));
	  V(1) = alpha*(-1*Cx(0,1) - Cy(1,1));
	  
	  
   }
};

class InvJac : public MatrixCoefficient
{
   private:
      GridFunction &phi; // vector-valued GridFunction
   public:
      InvJac(GridFunction &phi_) : MatrixCoefficient(phi_.FESpace()->GetVDim()), phi(phi_) {}

   virtual void Eval(DenseMatrix &M, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
	  int dim = phi.FESpace()->GetVDim();

      DenseMatrix Jac;
      phi.GetVectorGradient(T, Jac);  // Jac(i,j) = d phi_i / dx_j
	  Jac(0,0) += 1.0; Jac(1,1) += 1.0;
	  
	  M.SetSize(dim, dim);
	  M = Jac;
	  M.Invert();
   }
};

class RHSg : public Coefficient //Takes in two terms
{
   private:
      GridFunction &phi; // vector-valued GridFunction
	  GridFunction &phidot; // vector-valued GridFunction
   public:
      RHSg(GridFunction &phi_, GridFunction &phidot_) : phi(phi_), phidot(phidot_) {}

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
	   T.SetIntPoint(&ip);
	   int dim = phi.FESpace()->GetVDim();

      DenseMatrix JacInv(dim, dim), JacDot(dim, dim);
      phi.GetVectorGradient(T, JacInv); 
	  JacInv(0,0) += 1.0; JacInv(1,1) += 1.0; 
	  JacInv.Invert();
      phidot.GetVectorGradient(T, JacDot);

      DenseMatrix Mat(dim, dim), MatSqd(dim, dim);
      Mat = 0.0; AddMult(JacInv, JacDot, Mat);
      MatSqd = 0.0; AddMult(Mat, Mat, MatSqd);	  
	  
      return (Mat(0,0) + Mat(1,1))*(Mat(0,0) + Mat(1,1)) + (MatSqd(0,0) + MatSqd(1,1));
   }
};

class myGradScal : public VectorCoefficient
{
   private:
      int dim;
      GridFunction &x;   //scalar valued
   public:
      myGradScal(int dim_, GridFunction &x_) : VectorCoefficient(dim_), dim(dim_), x(x_) {}

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
      V.SetSize(dim);
      x.GetGradient(T, V);
   }
};

	

namespace hydrodynamics
{


MFEM_HOST_DEVICE inline double smooth_step_01(double x, double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

void LagrangianIGRHydroOperator::UpdateQuadratureDataIGR(const Vector &S) const
{
   if (qdata_is_current) { return; }

   qdata_is_current = true;
   forcemat_is_assembled = false;

   if (dim > 1 && p_assembly) { return qupdate->UpdateQuadratureData(S, qdata); }

   // This code is only for the 1D/FA mode
   LAGHOS_DEVICE_SYNC;
   timer.sw_qdata.Start();
   const int nqp = ir.GetNPoints();
   ParGridFunction x, v, e, igr;
   Vector* sptr = const_cast<Vector*>(&S);
   x.MakeRef(&H1, *sptr, 0);
   v.MakeRef(&H1, *sptr, H1.GetVSize());
   e.MakeRef(&L2, *sptr, 2*H1.GetVSize());
   igr.MakeRef(&fespace, *sptr, 2*H1.GetVSize() + L2.GetVSize());
   Vector e_vals;
   DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim);

   // Batched computations are needed, because hydrodynamic codes usually
   // involve expensive computations of material properties. Although this
   // miniapp uses simple EOS equations, we still want to represent the batched
   // cycle structure.
   int nzones_batch = 3;
   const int nbatches =  NE / nzones_batch + 1; // +1 for the remainder.
   int nqp_batch = nqp * nzones_batch;
   double *gamma_b = new double[nqp_batch],
   *rho_b = new double[nqp_batch],
   *e_b   = new double[nqp_batch],
   *p_b   = new double[nqp_batch],
   *cs_b  = new double[nqp_batch];
   // Jacobians of reference->physical transformations for all quadrature points
   // in the batch.
   DenseTensor *Jpr_b = new DenseTensor[nzones_batch];
   for (int b = 0; b < nbatches; b++)
   {
      int z_id = b * nzones_batch; // Global index over zones.
      // The last batch might not be full.
      if (z_id == NE) { break; }
      else if (z_id + nzones_batch > NE)
      {
         nzones_batch = NE - z_id;
         nqp_batch    = nqp * nzones_batch;
      }

      double min_detJ = std::numeric_limits<double>::infinity();
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1.GetElementTransformation(z_id);
         Jpr_b[z].SetSize(dim, dim, nqp);
         e.GetValues(z_id, ir, e_vals);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            Jpr_b[z](q) = T->Jacobian();
            const double detJ = Jpr_b[z](q).Det();
            min_detJ = fmin(min_detJ, detJ);
            const int idx = z * nqp + q;
            // Assuming piecewise constant gamma that moves with the mesh.
            gamma_b[idx] = gamma_gf(z_id);
            rho_b[idx] = qdata.rho0DetJ0w(z_id*nqp + q) / detJ / ip.weight;
            e_b[idx] = fmax(0.0, e_vals(q));
         }
         ++z_id;
      }

      // Batched computation of material properties.
      ComputeMaterialProperties(nqp_batch, gamma_b, rho_b, e_b, p_b, cs_b);

      z_id -= nzones_batch;
	  
	  //ParFiniteElementSpace fespace(pmesh, H1.FEColl(), 1, Ordering::byNODES);
	  //ParGridFunction IGRpressure(&fespace);
	  CalcIGRTerm(x, v, igr);
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1.GetElementTransformation(z_id);
         for (int q = 0; q < nqp; q++)
         {
            			
			
            // Note that the Jacobian was already computed above. We've chosen
            // not to store the Jacobians for all batched quadrature points.
            const DenseMatrix &Jpr = Jpr_b[z](q);
            CalcInverse(Jpr, Jinv);
            const double detJ = Jpr.Det(), rho = rho_b[z*nqp + q],
                         p = p_b[z*nqp + q], sound_speed = cs_b[z*nqp + q];
						
			//Set Stress + IGR pressure			
			const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
			
			//double IGR_Pressure = igr.GetValue(*T, ip); // New IGR term
			/*stress = 0.0;
			DenseMatrix J(dim);
            x.GetVectorGradient(*T, J);
			DenseMatrix JinvT(J);
            JinvT.Invert();     // JinvT now holds (Dx)^{-1}
            JinvT.Transpose();
			stress = JinvT;
			stress *= alpha;
			stress *= IGR_Pressure;
            for (int d = 0; d < dim; d++) { stress(d, d) -= p; }*/
            //for (int d = 0; d < dim; d++) { stress(d, d) = alpha*IGR_Pressure-p; }
			
			const double igr_p = igr.GetValue(*T, ip);

            // If you truly want (Dx)^{-T} scaling:
            DenseMatrix Dx(dim); x.GetVectorGradient(*T, Dx);
            DenseMatrix JinvT(Dx); JinvT.Invert(); JinvT.Transpose();

            //stress = JinvT;
            stress = 0.0; stress(0,0) = 1.0; stress(1,1) = 1.0; // Identity                   
            stress *= alpha * igr_p;            
            for (int d = 0; d < dim; d++) { stress(d,d) -= p; }
            

            double visc_coeff = 0.0;
            if (use_viscosity)
            {
               // Compression-based length scale at the point. The first
               // eigenvector of the symmetric velocity gradient gives the
               // direction of maximal compression. This is used to define the
               // relative change of the initial length scale.
               v.GetVectorGradient(*T, sgrad_v);

               double vorticity_coeff = 1.0;
               if (use_vorticity)
               {
                  const double grad_norm = sgrad_v.FNorm();
                  const double div_v = fabs(sgrad_v.Trace());
                  vorticity_coeff = (grad_norm > 0.0) ? div_v / grad_norm : 1.0;
               }

               sgrad_v.Symmetrize();
               double eig_val_data[3], eig_vec_data[9];
               if (dim==1)
               {
                  eig_val_data[0] = sgrad_v(0, 0);
                  eig_vec_data[0] = 1.;
               }
               else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
               Vector compr_dir(eig_vec_data, dim);
               // Computes the initial->physical transformation Jacobian.
               mfem::Mult(Jpr, qdata.Jac0inv(z_id*nqp + q), Jpi);
               Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
               // Change of the initial mesh size in the compression direction.
               const double h = qdata.h0 * ph_dir.Norml2() /
                                compr_dir.Norml2();
               // Measure of maximal compression.
               const double mu = eig_val_data[0];
               visc_coeff = 2.0 * rho * h * h * fabs(mu);
               // The following represents a "smooth" version of the statement
               // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
               // eps must be scaled appropriately if a different unit system is
               // being used.
               const double eps = 1e-12;
               visc_coeff += 0.5 * rho * h * sound_speed * vorticity_coeff *
                             (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
               stress.Add(visc_coeff, sgrad_v);
            }
            // Time step estimate at the point. Here the more relevant length
            // scale is related to the actual mesh deformation; we use the min
            // singular value of the ref->physical Jacobian. In addition, the
            // time step estimate should be aware of the presence of shocks.
            const double h_min =
               Jpr.CalcSingularvalue(dim-1) / (double) H1.GetOrder(0);
            const double inv_dt = sound_speed / h_min +
                                  2.5 * visc_coeff / rho / h_min / h_min;
            if (min_detJ < 0.0)
            {
               // This will force repetition of the step with smaller dt.
               qdata.dt_est = 0.0;
            }
            else
            {
               if (inv_dt>0.0)
               {
                  qdata.dt_est = fmin(qdata.dt_est, cfl*(1.0/inv_dt));
               }
            }
            // Quadrature data for partial assembly of the force operator.
            MultABt(stress, Jinv, stressJiT);
            stressJiT *= ir.IntPoint(q).weight * detJ;
            for (int vd = 0 ; vd < dim; vd++)
            {
               for (int gd = 0; gd < dim; gd++)
               {
                  qdata.stressJinvT(vd)(z_id*nqp + q, gd) =
                     stressJiT(vd, gd);
               }
            }
         }
         ++z_id;
      }
   }
   delete [] gamma_b;
   delete [] rho_b;
   delete [] e_b;
   delete [] p_b;
   delete [] cs_b;
   delete [] Jpr_b;
   LAGHOS_DEVICE_SYNC;
   timer.sw_qdata.Stop();
   timer.quad_tstep += NE;
}






void LagrangianIGRHydroOperator::CalcIGRTerm(ParGridFunction &Phi, ParGridFunction &PhiDot, ParGridFunction &x) const
{
   ParFiniteElementSpace fespace(pmesh, H1.FEColl(), 1, Ordering::byNODES);
   //  Here I replaced feVECspace with the existing H1
   
   Array<int> ess_tdof_list;


   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   //int nv = pmesh.GetNV();
   //ParFiniteElementSpace feVECspace(&pmesh, fec, dim);
   //ParGridFunction Phi(&feVECspace), PhiDot(&feVECspace);
      
   
   
   RHSg gCoeff(Phi, PhiDot);
   
   b.AddDomainIntegrator(new DomainLFIntegrator(gCoeff));
   b.Assemble();

   //ParGridFunction x(&fespace);
   x = 0.0;

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new MassIntegrator); 
   MassMatrix1 M(Phi, alpha);
   a.AddDomainIntegrator(new DiffusionIntegrator(M));



   //Calculate the gradients of each component of Phi
   //Split phi into components
   Vector e0(2); e0(0) = 1.0; e0(1) = 0.0;
   Vector e1(2); e1(0) = 0.0; e1(1) = 1.0;
   VectorConstantCoefficient e0Coeff(e0), e1Coeff(e1);
   VectorGridFunctionCoefficient PhiCoeff(&Phi);
   InnerProductCoefficient comp0(PhiCoeff, e0Coeff), comp1(PhiCoeff, e1Coeff);
   ParGridFunction comp0grid(&fespace), comp1grid(&fespace);
   comp0grid.ProjectCoefficient(comp0);
   comp1grid.ProjectCoefficient(comp1);

   //Each scalar piece has a gradient
   GradientGridFunctionCoefficient gradX(&comp0grid);
   GradientGridFunctionCoefficient gradY(&comp1grid);
   ParGridFunction gradXGrid(&H1), gradYGrid(&H1);
   gradXGrid.ProjectCoefficient(gradX);
   gradYGrid.ProjectCoefficient(gradY);
   
   
   LambdaDivPart Lpt1(Phi, gradXGrid, gradYGrid, sqrt(alpha));                    //Custom Vector Coefficient
   InnerProductCoefficient Lambda1(Lpt1, Lpt1);             //This is a scalar coefficient
   a.AddDomainIntegrator(new MassIntegrator(Lambda1));
   
   
   LambdaDivPart Lpt2(Phi, gradXGrid, gradYGrid, alpha);
   InvJac DPhiInv(Phi);                                     //Computes the inverse Jacobian of Phi
   MatrixVectorProductCoefficient Lambda2(DPhiInv, Lpt2);   //This is a vector coefficient
   a.AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(Lambda2));
   a.AddDomainIntegrator(new TransposeIntegrator(new MixedDirectionalDerivativeIntegrator(Lambda2)));

   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // 11. Solve the linear system A X = B.
   Solver *prec = NULL;
   //prec = new HypreBoomerAMG;
   //((HypreBoomerAMG *)prec)->SetPrintLevel(0);
   //prec = new HypreIdentity;
   HypreSmoother M_prec;
   M_prec.SetType(HypreSmoother::Jacobi);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(0);
   if (true) { cg.SetPreconditioner(M_prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   //delete prec;

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);
   x *= -1.0;
   
}










} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

