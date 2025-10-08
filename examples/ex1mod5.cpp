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

class ReducedSystemOperator;

class IGROperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   FiniteElementSpace &feVECspace;
   Mesh &mesh; 


   //BilinearForm M, S;
   //NonlinearForm H;
   //real_t viscosity;
   //HyperelasticModel *model;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   /** Nonlinear operator defining the reduced backward Euler equation for the
       velocity. Used in the implementation of method ImplicitSolve. */
   //ReducedSystemOperator *reduced_oper;

   
   NewtonSolver newton_solver; /// Newton solver for the reduced backward Euler equation

   Solver *J_solver; /// Solver for the Jacobian solve in the Newton method
   Solver *J_prec; /// Preconditioner for the Jacobian solve in the Newton method

   mutable Vector z; // auxiliary vector

public:
   IGROperator(FiniteElementSpace &fscal, FiniteElementSpace &fvec, Mesh &mesh);

   /// Compute the right-hand side of the ODE system.
   void Mult(const Vector &vx, Vector &dvx_dt) const override;
 
   ~IGROperator() override;
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   //const char *mesh_file = "../data/periodic-square.mesh";
   int order = 1;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   real_t dt = 0.0001;
   real_t t_final = 0.01;
   real_t alpha = 0.01;
   int ode_solver_type = 2;


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
         (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the external boundary attributes from the mesh as essential (Dirichlet)
   //    and converting them to a list of true dofs.
   

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   int nv = mesh.GetNV();
   
   FiniteElementSpace feVECspace(&mesh, fec, dim);
   GridFunction Phi(&feVECspace), PhiDot(&feVECspace);
   

   //Different Initial condition templates
   VectorFunctionCoefficient identity(mesh.Dimension(),
    [](const Vector &x, Vector &y) { y = x; });
   VectorFunctionCoefficient zerofunc(mesh.Dimension(),
    [](const Vector &x, Vector &y) { y = 0.0; });   
   VectorFunctionCoefficient bump(mesh.Dimension(),
    [](const Vector &x, Vector &y) { 
	  double width = 0.2;
	  if(abs(x[0] - 0.5) < width && abs(x[1] - 0.5) < width){
		  y[0] = (width-x[0] + 0.5)*0.02;
		  y[1] = (width-x[1] + 0.5)*0.02;
	  } else {
		  y = 0.0;
	  }
	}); 
   VectorFunctionCoefficient shock(mesh.Dimension(),
    [](const Vector &x, Vector &y) { 
	  if(x[0] < 0){
		  y[0] = 1.0;
		  y[1] = 0.0;
	  } else {
		  y = 0.0;
	  }
	}); 
   VectorFunctionCoefficient smooth(mesh.Dimension(),
   [](const Vector &x, Vector &y) {
    y = 0.0;
    y[0] = exp(-40*(pow(x[0]-0.5,2) + pow(x[1]-0.5,2)));
   });
   VectorFunctionCoefficient shock2(mesh.Dimension(),
    [](const Vector &x, Vector &y) { 
	  if(x[0] < 0.35){
		  y[0] = 1.0;
		  y[1] = 0.0;
	  } else if(x[0] < 0.45){
		  y[0] = 4.5 - 10*x[0];
		  y[1] = 0.0;
	  } else {
		  y = 0.0;
	  }
	}); 
   
	
   Phi = 0.0;
   GridFunction Phidotdotgf(&feVECspace);  // same order as fespace


   //New compared to ex1mod4:
   //Here try to set up time dependence/ time integrator
   int fe_size = feVECspace.GetTrueVSize();
   Array<int> fe_offset(3);
   fe_offset[0] = 0;   fe_offset[1] = fe_size; fe_offset[2] = 2*fe_size;
   BlockVector vx(fe_offset);


   // bind
   PhiDot.MakeTRef(&feVECspace, vx.GetBlock(0), 0);
   Phi.MakeTRef(&feVECspace, vx.GetBlock(1), 0);

   PhiDot.ProjectCoefficient(shock2);

   socketstream sol_sock;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sol_sock.open(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << PhiDot << flush;
   }

   // after binding, vx block still zero
   cout << "vx block 0 norm after bind = " << vx.GetBlock(0).Norml2() << endl;



   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
   
   real_t t = 0.0;
   IGROperator oper(fespace, feVECspace, mesh);
   oper.SetTime(t);
   ode_solver->Init(oper);
   
   //Time integration

   bool last_step = false;
   for(int ti =0; !last_step; ti++){
      
      real_t dt_real = min(dt, t_final - t);
      
      ode_solver->Step(vx, t, dt_real);

      last_step = (t >= t_final - 1e-8*dt);

      if (visualization) // every step
      {
         sol_sock << "solution\n" << mesh << PhiDot << flush;
      }
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}



IGROperator::IGROperator(FiniteElementSpace &fscal, FiniteElementSpace &fvec, Mesh &mesh)
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

   cout << "sc = " << sc
     << ", ||v|| = " << v.Norml2()
     << ", ||x|| = " << x.Norml2() << endl;

   // Wrap x into GridFunctions (no copies, just views)
   GridFunction Phi(&feVECspace), PhiDot(&feVECspace);
   //Phi.MakeRef(&feVECspace, x, 0);
   //PhiDot.MakeRef(&feVECspace, v, 0);
   Phi.MakeRef(&feVECspace, const_cast<Vector&>(vx), sc);
   PhiDot.MakeRef(&feVECspace, const_cast<Vector&>(vx), 0);

   // Wrap dxdt into GridFunctions
   GridFunction ddphi;
   ddphi.MakeRef(&feVECspace, dx_dt, 0);


   //Copied one-step calculation
   Array<int> ess_tdof_list;


   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   int nv = mesh.GetNV();
   int dim = 2;
   
   RHSg gCoeff(Phi, PhiDot);
   
   b.AddDomainIntegrator(new DomainLFIntegrator(gCoeff));
   b.Assemble();

   //    Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x2(&fespace);
   x2 = 0.0;

   //    Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm a(&fespace);
   
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
   GridFunction comp0grid(&fespace), comp1grid(&fespace);
   comp0grid.ProjectCoefficient(comp0);
   comp1grid.ProjectCoefficient(comp1);

   //Each scalar piece has a gradient
   GradientGridFunctionCoefficient gradX(&comp0grid);
   GradientGridFunctionCoefficient gradY(&comp1grid);
   GridFunction gradXGrid(&feVECspace), gradYGrid(&feVECspace);
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

   //     Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   a.Assemble();
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x2, b, A, X, B);

   // Solve the linear system A X = B.
   #ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother Msmooth((SparseMatrix&)(*A));
      PCG(*A, Msmooth, B, X, 1, 400, 1e-12, 0.0);
   #endif



   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x2);
   //x2 is the result of the elliptic solve
   //Want Phi'' = div([DPhi]^-T x2 + Euler Term )
   
   //Recover Phi''
   TransposeMatrixCoefficient DPhiInvT(DPhiInv);
   myGradScal Dx(dim, x2);
   MatrixVectorProductCoefficient FirstProductRulePt(DPhiInvT, Dx);
   
   LambdaDivPart DivDPhiInvT(Phi, gradXGrid, gradYGrid, 1.0);
   GridFunctionCoefficient xcoeff(&x2);  
   ScalarVectorProductCoefficient SecondProductRulePt(xcoeff, DivDPhiInvT);
   
   VectorSumCoefficient Phidotdotcoeff(FirstProductRulePt, SecondProductRulePt);
   
   
   ScalarVectorProductCoefficient negCoeff(-1.0, Phidotdotcoeff); //Fix sign?
   
   GridFunction Phidotdotgf(&feVECspace);  // same order as fespace
   Phidotdotgf.ProjectCoefficient(negCoeff);
   

   //Set Outputs
   dv_dt = Phidotdotgf; // + div()
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

