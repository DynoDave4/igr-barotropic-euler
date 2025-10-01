//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/octahedron.mesh -o 1
//               ex1 -m ../data/periodic-annulus-sector.msh
//               ex1 -m ../data/periodic-torus-sector.msh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -fa -d cuda
//               ex1 -pa -d raja-cuda
//             * ex1 -pa -d raja-hip
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cpu -o 4 -a
//               ex1 -pa -d ceed-cpu -m ../data/square-mixed.mesh
//               ex1 -pa -d ceed-cpu -m ../data/fichera-mixed.mesh
//             * ex1 -pa -d ceed-cuda
//             * ex1 -pa -d ceed-hip
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cpu
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Poisson problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

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
	  float alpha;
   public:
      MassMatrix1(GridFunction &phi_, float alpha_) : MatrixCoefficient(phi_.FESpace()->GetVDim()), phi(phi_), alpha(alpha_) {}

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
	  float alpha;
   public:
      LambdaDivPart(GridFunction &phi_, GridFunction &gradX_, GridFunction &gradY_, float alpha_) : 
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
      myGradScal(int dim_, GridFunction &x_) : VectorCoefficient(dim), dim(dim_), x(x_) {}

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
      V.SetSize(dim);
      x.GetGradient(T, V);
   }
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
   bool algebraic_ceed = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
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
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the external boundary attributes from the mesh as essential (Dirichlet)
   //    and converting them to a list of true dofs.
   Array<int> ess_tdof_list;

   //Start Timer
   auto start = chrono::high_resolution_clock::now();

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   int nv = pmesh.GetNV();
   
   ParFiniteElementSpace feVECspace(&pmesh, fec, dim);
   ParGridFunction Phi(&feVECspace), PhiDot(&feVECspace);
   
   VectorFunctionCoefficient identity(pmesh.Dimension(),
    [](const Vector &x, Vector &y) { y = x; });
   VectorFunctionCoefficient zerofunc(pmesh.Dimension(),
    [](const Vector &x, Vector &y) { y = 0.0; });   
   VectorFunctionCoefficient bump(pmesh.Dimension(),
    [](const Vector &x, Vector &y) { 
	  float width = 0.2;
	  if(abs(x[0]) < width && abs(x[1]) < width){
		  y[0] = (width-abs(x[0]))*(width-abs(x[0]))*0.01;
		  y[1] = (width-abs(x[1]))*(width-abs(x[1]))*0.01;
	  } else {
		  y = 0.0;
	  }
	});
   VectorFunctionCoefficient shock(pmesh.Dimension(),
    [](const Vector &x, Vector &y) { 
	  float width = 0.2;
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
    y[0] = exp(-40*(pow(x[0]-0.5,2) + pow(x[1]-0.5,2)));
   });
   
	
   Phi = 0.0;
   PhiDot.ProjectCoefficient(shock); 
      
   
   
   RHSg gCoeff(Phi, PhiDot);
   
   b.AddDomainIntegrator(new DomainLFIntegrator(gCoeff));
   //b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ParBilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   
   float alpha = 0.1; 
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

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   Solver *prec = NULL;
   if (pa)
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            prec = new ceed::AlgebraicSolver(a, ess_tdof_list);
         }
         else
         {
            prec = new OperatorJacobiSmoother(a, ess_tdof_list);
         }
      }
   }
   else
   {
      prec = new HypreBoomerAMG;
   }
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);
   
   /*for (int vi = 0; vi < 30; vi++)
   {
	const double *v = pmesh.GetVertex(vi); // pointer to coords
    cout << "Vertex " << vi << " : (" << v[0] << ", " << v[1];
	cout << ") and here x(" << vi << ") is: (" << x(feVECspace.DofToVDof(vi, 0)) << 
	", " << x(feVECspace.DofToVDof(vi, 0)+nv) << ") \n";
   }*/
   


   //End timer
   auto end = chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;
   double elapsed_seconds = duration.count();
   cout << "Execution time: " << elapsed_seconds << " seconds." << endl;
   
   //Recover Phi''
   TransposeMatrixCoefficient DPhiInvT(DPhiInv);
   myGradScal Dx(dim, x);
   MatrixVectorProductCoefficient FirstProductRulePt(DPhiInvT, Dx);
   
   LambdaDivPart DivDPhiInvT(Phi, gradXGrid, gradYGrid, 1.0);
   GridFunctionCoefficient xcoeff(&x);  
   ScalarVectorProductCoefficient SecondProductRulePt(xcoeff, DivDPhiInvT);
   
   VectorSumCoefficient Phidotdotcoeff(FirstProductRulePt, SecondProductRulePt);
   
   
   ScalarVectorProductCoefficient negCoeff(-1.0, Phidotdotcoeff);
   
   ParGridFunction Phidotdotgf(&feVECspace);  // same order as fespace
   Phidotdotgf.ProjectCoefficient(negCoeff);
   

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   InnerProductCoefficient ddcomp0(Phidotdotcoeff, e0Coeff);
   ParGridFunction ddcomp0grid(&fespace);
   ddcomp0grid.ProjectCoefficient(ddcomp0);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
