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

double f_fun(const Vector &x)
   {
      return x[0]*x[0] + x[1]*x[1] + x[0]*x[0]*x[0]*x[0]; // works in 2D
	  //return 2*x[0];
   }

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
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
   FiniteElementSpace feVECspace(&mesh, fec, dim, Ordering::byNODES);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the external boundary attributes from the mesh as essential (Dirichlet)
   //    and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      // Apply boundary conditions on all external boundaries:
      mesh.MarkExternalBoundaries(ess_bdr);
      // Boundary conditions can also be applied based on named attributes:
      // mesh.MarkNamedBoundaries(set_name, ess_bdr)

      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   cout << "vdim = " << feVECspace.GetVDim() << endl;
   
   
   // Initialize function to take the gradient of
   FunctionCoefficient f_coeff(f_fun);
   
   ConstantCoefficient one(1.0);
   GridFunction f(&fespace);
   f.ProjectCoefficient(f_coeff);
   
   //Print f values
   for (int vi = 0; vi < 10; ++vi)
   {
    const double *v = mesh.GetVertex(vi); // pointer to coords
    std::cout << "Vertex " << vi << " : (";
    for (int d = 0; d < dim; ++d)
    {
        if (d) std::cout << ", ";
        std::cout << v[d];
    }
	Array<int> vdofs;
    feVECspace.GetVertexVDofs(vi, vdofs);

    // now extract values from x at those dofs
    Vector val(vdofs.Size());
    f.GetSubVector(vdofs, val);
	std::cout << ") f = x^2 + y^2: = (" << val(0) << ") \n";
   }   
   

   GradientGridFunctionCoefficient grad(&f);
   
   //Start Timer
   auto start = std::chrono::high_resolution_clock::now();
   
   LinearForm b(&feVECspace);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(grad));
   b.Assemble();

   //    Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&feVECspace);
   x = 0.001;

   //    Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm a(&feVECspace);
   a.AddDomainIntegrator(new VectorMassIntegrator(one));
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // Solve the linear system A X = B.
   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 1, 200, 1e-25, 0.0);


   // Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);
   
   int nv = mesh.GetNV();
   const int component = 0, component2 = 1;
   cout << "There are " << nv << " vertices in the mesh. \n ";
   
   // End timer
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;
   double elapsed_seconds = duration.count();
   cout << "\n Attempt 1 setting up the linear system does not get the first coord right. ";
   std::cout << "Execution time: " << elapsed_seconds << " seconds." << std::endl;

   
   //Print Gradient
   for (int vi = 0; vi < 10; vi++)
   {
	const double *v = mesh.GetVertex(vi); // pointer to coords
    cout << "Vertex " << vi << " : (" << v[0] << ", " << v[1];
	cout << ") and here x(" << vi << ") is: (" << x(feVECspace.DofToVDof(vi, 0)) << 
	", " << x(feVECspace.DofToVDof(vi, 1)) << ") \n";
   }
   
  
   //Attempt 2 Fails 
   
   start = chrono::high_resolution_clock::now();
   GridFunction x0(&fespace), x1(&fespace);
   
   //f.GetDerivative(0, 0, x0);
   //f.GetDerivative(0, 1, x1);
   
   end = chrono::high_resolution_clock::now();
   duration = end - start;
   elapsed_seconds = duration.count();
   cout << "\n Attempt 2 using GetDerivative does not work. ";
   cout << "Execution time: " << elapsed_seconds << " seconds." << endl;
   
   
   //Print Gradient
   Vector nodal_vals0, nodal_vals1;
   x0.GetNodalValues(nodal_vals0);
   x1.GetNodalValues(nodal_vals1);
   
   //Attempt 3 - Just project it  - Not working amazingly, but mostly correct
   start = chrono::high_resolution_clock::now();
   
   //Recall:    GridFunction f(&fespace);
   //           GradientGridFunctionCoefficient grad(&f);
   GridFunction gradgrid(&feVECspace);
   gradgrid.ProjectCoefficient(grad);   
   
   end = chrono::high_resolution_clock::now();
   duration = end - start;
   elapsed_seconds = duration.count();
   cout << "\n Attempt 3 just projects the gradientgridfunction. ";
   cout << "Execution time: " << elapsed_seconds << " seconds." << endl;

   //Print Gradient
   for (int vi = 0; vi < 10; vi++)
   {
	const double *v = mesh.GetVertex(vi); // pointer to coords
    cout << "Vertex " << vi << " : (" << v[0] << ", " << v[1];
	cout << ") and here gradgrid(" << vi << ") is: (" << gradgrid(feVECspace.DofToVDof(vi, 0)) << 
	", " << gradgrid(feVECspace.DofToVDof(vi, 0)+nv) << ") \n";
   }
   
   
   // Attempt 4

   
   //Begin hessian calculations. Remember, we have the initial grid function f,
   // the results ofthe linear solve for the gradient for attempt 1 are stored in x
   

   
   start = std::chrono::high_resolution_clock::now();
   
   //Recall:    GridFunction f(&fespace);
   //           GradientGridFunctionCoefficient grad(&f);
   //Array<int> indicesNV(nv), indicesNV2(nv); 
   //iota(indicesNV.begin(), indicesNV.end(), 0);     //Indices (0 -> nv-1)
   //iota(indicesNV2.begin(), indicesNV2.end(), nv);  //Indices (nv -> 2nv-1)
   
   //Vector temp0(nv), temp1(nv);                     //The different components of the gradient grid function
   //gradgrid.GetSubVector(indicesNV, temp0);
   //gradgrid.GetSubVector(indicesNV2, temp1);
   
   Array<int> vdofs;   
   GridFunction gradcomp0(&fespace), gradcomp1(&fespace);

   // Loop over all scalar dofs
   
   for (int i = 0; i < nv; i++)
   {
      feVECspace.GetVertexVDofs(i, vdofs); // gives indices of the vector dofs for this scalar dof
	  //gradcomp0(i) = gradgrid(vdofs[0]);
	  //gradcomp1(i) = gradgrid(vdofs[1]); 
	  gradcomp0(i) = gradgrid(feVECspace.DofToVDof(i, 0));
	  gradcomp1(i) = gradgrid(feVECspace.DofToVDof(i, 1));
	  
      //gradcomp1(i) = gradgrid(vdofs[1]);
   }
   

   
   
   GradientGridFunctionCoefficient hessX(&gradcomp0), hessY(&gradcomp1);  //Store vectors of the hessian
   GridFunction hessXgrid(&feVECspace), hessYgrid(&feVECspace);
   hessXgrid.ProjectCoefficient(hessX);
   hessYgrid.ProjectCoefficient(hessY);    
   
   end = std::chrono::high_resolution_clock::now();
   duration = end - start;
   elapsed_seconds = duration.count();
   
   
   //Check components
   cout << "Check the components. \n ";
   for (int vi = 0; vi < 10; vi++)
   {
	const double *v = mesh.GetVertex(vi); // pointer to coords
    cout << "Vertex " << vi << " : (" << v[0] << ", " << v[1];
	cout << ") and here grad components are (" << gradcomp0(vi) << ", " << gradcomp1(vi) << ") \n";
   }
   
   cout << "\n This is the Hessian calculation. ";
   std::cout << "Execution time: " << elapsed_seconds << " seconds." << std::endl;
   
   
   //Print Hessian
   for (int vi = 0; vi < 10; vi++)
   {
	const double *v = mesh.GetVertex(vi); // pointer to coords
    cout << "Vertex " << vi << " - Hessian(" << v[0] << ", " << v[1];
	cout << ") = (" << hessXgrid(vi) << ", " << hessXgrid(vi+nv) << "; ";
	cout << hessYgrid(2*vi) << ", " << hessYgrid(2*vi+1) << ") \n";
   }
   
   
   
   
   //Hessian attempt 2
   start = chrono::high_resolution_clock::now();
   
   //Recall:    GridFunction f(&fespace);
   //           GradientGridFunctionCoefficient grad(&f);
   Vector e0(2); e0(0) = 1.0; e0(1) = 0.0;
   Vector e1(2); e1(0) = 0.0; e1(1) = 1.0;
   VectorConstantCoefficient e0Coeff(e0), e1Coeff(e1);
   InnerProductCoefficient grad0(grad, e0Coeff), grad1(grad, e1Coeff);
   
   GridFunction gradgrid0(&fespace), gradgrid1(&fespace);
   gradgrid0.ProjectCoefficient(grad0);
   gradgrid1.ProjectCoefficient(grad1);
   
   GradientGridFunctionCoefficient HessRow0(&gradgrid0), HessRow1(&gradgrid1);
   GridFunction HessRow0Grid(&feVECspace), HessRow1Grid(&feVECspace);
   HessRow0Grid.ProjectCoefficient(HessRow0);
   HessRow1Grid.ProjectCoefficient(HessRow1);   
   
   end = chrono::high_resolution_clock::now();
   duration = end - start;
   elapsed_seconds = duration.count();
   
   //Check components 2
   cout << "Check the components. \n ";
   for (int vi = 0; vi < 10; vi++)
   {
	const double *v = mesh.GetVertex(vi); // pointer to coords
    cout << "Vertex " << vi << " : (" << v[0] << ", " << v[1];
	cout << ") and here gradgrid components are (" << gradgrid0(vi) << ", " << gradgrid1(vi) << ") \n";
   }
   
   
   cout << "\n Hessian Attempt 2 ";
   cout << "Execution time: " << elapsed_seconds << " seconds." << endl;
   
   //Print Hessian
   for (int vi = 0; vi < 10; vi++)
   {
	const double *v = mesh.GetVertex(vi); // pointer to coords
    cout << "Vertex " << vi << " - Hessian(" << v[0] << ", " << v[1] << ") = (" << 
	HessRow0Grid(feVECspace.DofToVDof(vi, 0)) << ", " << 
	HessRow0Grid(feVECspace.DofToVDof(vi, 1)) << "; " << 
	HessRow1Grid(feVECspace.DofToVDof(vi, 0)) << ", " << 
	HessRow1Grid(feVECspace.DofToVDof(vi, 1)) << ") \n";
   }
   
   
   
   
   
   

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
