//                                IGR Shock Attempt 1
//

//
// Description:  This code is a first (attempt at) working code to 
//               regularize a shock using IGR.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

using namespace std;
using namespace mfem;

int debug=1;

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

class IGROperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   ParFiniteElementSpace &feVECspace;
   ParMesh &mesh; 


   //BilinearForm M, S;
   //NonlinearForm H;
   //real_t viscosity;
   //HyperelasticModel *model;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   /** Nonlinear operator defining the reduced backward Euler equation for the
       velocity. Used in the implementation of method ImplicitSolve. */
   
   NewtonSolver newton_solver; /// Newton solver for the reduced backward Euler equation

   Solver *J_solver; /// Solver for the Jacobian solve in the Newton method
   Solver *J_prec; /// Preconditioner for the Jacobian solve in the Newton method

   mutable Vector z; // auxiliary vector

public:
   IGROperator(ParFiniteElementSpace &fscal, ParFiniteElementSpace &fvec, ParMesh &mesh);

   /// Compute the right-hand side of the ODE system.
   void Mult(const Vector &vx, Vector &dvx_dt) const override;
 
   ~IGROperator() override;
};


int main(int argc, char *argv[])
{
   //0. Initialize MPI and HYPRE
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   //const char *mesh_file = "../data/periodic-square.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   real_t dt = 0.0001;
   real_t t_final = 0.01;
   real_t alpha = 0.01;
   int ode_solver_type = 1;
   real_t refined = 10000.0;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&dt, "-dt", "--timestep",
                  "The time step of the method");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Alpha as the level of IGR");
   args.AddOption(&t_final, "-tf", "--final-time",
                  "The ending time");
   args.AddOption(&refined, "-rf", "--init-max-refinement",
                  "The ending time");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);


   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(refined/mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 4b. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   bool delete_fec = true;
   H1_FECollection fec(order, dim);

   ParFiniteElementSpace fespace(&pmesh, &fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }
   
   ParFiniteElementSpace feVECspace(&pmesh, &fec, dim);

   //Different Initial condition templates
   VectorFunctionCoefficient identity(pmesh.Dimension(),
    [](const Vector &x, Vector &y) { y = x; });
   VectorFunctionCoefficient zerofunc(pmesh.Dimension(),
    [](const Vector &x, Vector &y) { y = 0.0; });   
   VectorFunctionCoefficient bump(pmesh.Dimension(),
    [](const Vector &x, Vector &y) { 
	  double width = 0.2;
	  if(abs(x[0] - 0.5) < width && abs(x[1] - 0.5) < width){
		  y[0] = (width-x[0] + 0.5)*0.02;
		  y[1] = (width-x[1] + 0.5)*0.02;
	  } else {
		  y = 0.0;
	  }
	}); 
   VectorFunctionCoefficient shock(pmesh.Dimension(),
    [](const Vector &x, Vector &y) { 
	  if(x[0] < 0){
		  y[0] = 1.0;
		  y[1] = 0.0;
	  } else {
		  y = 0.0;
	  }
	}); 
   VectorFunctionCoefficient smooth(pmesh.Dimension(),
   [](const Vector &x, Vector &y) {
    y = 0.0;
    y[0] = exp(-40.0*(pow(x[0]-0.5,2) + pow(x[1]-0.5,2)));
   });
   VectorFunctionCoefficient shock2(pmesh.Dimension(),
    [](const Vector &x, Vector &y) { 
	  if(x[0] < 0.35){
		  y[0] = 1.0;
		  y[1] = 0.0;
	  } else if(x[0] < 0.45){
		  y[0] = 4.5 - 10.0*x[0];
		  y[1] = 0.0;
	  } else {
		  y = 0.0;
	  }
	}); 

   if(debug)
   {
     ostringstream mesh_name;
     mesh_name << "mesh." << setfill('0') << setw(6) << myid;
     ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     pmesh.Print(mesh_ofs);
   } 
	
   //New compared to ex1mod4:
   //Here try to set up time dependence/ time integrator
   int true_size = feVECspace.GetTrueVSize();
   Array<int> true_offset(3);
   true_offset[0] = 0;   true_offset[1] = true_size; true_offset[2] = 2*true_size;
   BlockVector vx(true_offset);
   ParGridFunction Phi(&feVECspace), PhiDot(&feVECspace);

   PhiDot.MakeTRef(&feVECspace, vx, true_offset[0]);
   Phi.MakeTRef(&feVECspace, vx, true_offset[1]);

   Phi = 0.0;
   PhiDot.ProjectCoefficient(shock2);

   PhiDot.GetTrueDofs(vx.GetBlock(0));
   Phi.GetTrueDofs(vx.GetBlock(1));

   //cout << "vx block0 size = " << vx.GetBlock(0).Size()  << ", PhiDot true size = " << feVECspace.GetTrueVSize()  << ", PhiDot vsize = " << feVECspace.GetVSize() << endl;

   

   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sol_sock.open(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << PhiDot << flush;
   }


   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
   
   real_t t = 0.0;
   IGROperator oper(fespace, feVECspace, pmesh);
   oper.SetTime(t);
   ode_solver->Init(oper);
   
   //Time integration

   bool last_step = false;


   for(int ti =0; !last_step; ti++){
      
      real_t dt_real = min(dt, t_final - t);
      
      ode_solver->Step(vx, t, dt_real);
      

      last_step = (t >= t_final - 1e-8*dt);

      if (visualization)
      {
         PhiDot.SetFromTrueDofs(vx.GetBlock(0));   // update local storage from true-dof state
         // Optionally also update displacement:
         Phi.SetFromTrueDofs(vx.GetBlock(1));
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << PhiDot << flush;
      }

      
   }

   

   

   // 15. Free the used memory.
   if (delete_fec)
   {
      //delete fec;
   }

   return 0;
}



IGROperator::IGROperator(ParFiniteElementSpace &fscal, ParFiniteElementSpace &fvec, ParMesh &mesh)
   : TimeDependentOperator(2*fvec.GetTrueVSize(), (real_t) 0.0), fespace(fscal), feVECspace(fvec), mesh(mesh)
{



}

void IGROperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   /*cout << "sc = " << sc
     << ", ||v|| = " << v.Norml2()
     << ", ||x|| = " << x.Norml2() << endl;*/


   // Quick sanity checks
   MFEM_ASSERT(sc == feVECspace.GetTrueVSize(), "sc mismatch with feVECspace true size");
   if (!IsFinite(v.Norml2()) || !IsFinite(x.Norml2()))
   {
      cout << "Rank ??? warning: input true-dof norms not finite: "
           << "||v_true||=" << v.Norml2() << " ||x_true||=" << x.Norml2() << endl;
   }


   // Wrap x into GridFunctions (no copies, just views)
   ParGridFunction Phi(&feVECspace), PhiDot(&feVECspace);
   Phi.SetFromTrueDofs(x);
   PhiDot.SetFromTrueDofs(v);

   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   int dim = 2;
   
   RHSg gCoeff(Phi, PhiDot);
   
   b.AddDomainIntegrator(new DomainLFIntegrator(gCoeff));
   b.Assemble();

   ParBilinearForm a(&fespace);
   
   double alpha = 0.1; 
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
   ParGridFunction gradXGrid(&feVECspace), gradYGrid(&feVECspace);
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
   a.Finalize();
   HypreParMatrix *A = a.ParallelAssemble();

   Vector B(fespace.TrueVSize()), X(fespace.TrueVSize());
   X=0.0;
   b.ParallelAssemble(B);

   // Solve the linear system A X = B.
   Solver *prec = new HypreBoomerAMG;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   //build x2 from TrueDofs X
   ParGridFunction x2(&fespace);
   x2.SetFromTrueDofs(X);

   if(debug)
   {
     int myid = Mpi::WorldRank();
     ostringstream sol_name;
     sol_name << "x2." << setfill('0') << setw(6) << myid;
     ofstream sol_ofs(sol_name.str().c_str());
     sol_ofs.precision(8);
     x2.Save(sol_ofs);
   }
   
   //Recover Phi''
   TransposeMatrixCoefficient DPhiInvT(DPhiInv);
   myGradScal Dx(dim, x2);
   MatrixVectorProductCoefficient FirstProductRulePt(DPhiInvT, Dx);
   
   LambdaDivPart DivDPhiInvT(Phi, gradXGrid, gradYGrid, 1.0);
   GridFunctionCoefficient xcoeff(&x2);  
   ScalarVectorProductCoefficient SecondProductRulePt(xcoeff, DivDPhiInvT);
   
   VectorSumCoefficient Phidotdotcoeff(FirstProductRulePt, SecondProductRulePt);
   
   
   ScalarVectorProductCoefficient negCoeff(-1.0, Phidotdotcoeff); //Fix sign?
   
   ParGridFunction Phidotdotgf(&feVECspace);  // same order as fespace
   Phidotdotgf.ProjectCoefficient(negCoeff);
   

   //Set Outputs
   //Phidotdotgf.Distribute();
   Phidotdotgf.GetTrueDofs(dv_dt);
   dx_dt = v;

   cout << "||v|| = " << v.Norml2()
     << ", ||dv_dt|| = " << dv_dt.Norml2()
     << ", ||dx_dt|| = " << dx_dt.Norml2() << endl;
}


IGROperator::~IGROperator()
{
   delete J_solver;
   delete J_prec;
   //delete reduced_oper;
   //delete model;
}

