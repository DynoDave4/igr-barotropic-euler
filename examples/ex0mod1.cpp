//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Poisson
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class GradientSquaredTraceCoefficient : public Coefficient
   {
      private:
         const GridFunction &u;
         mutable mfem::DenseMatrix grad;

      public:
      GradientSquaredTraceCoefficient(const GridFunction &u_) : u(u_) {}

      virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
      {
         T.SetIntPoint(&ip);
         u.GetVectorGradient(T, grad);  // du = grad(u) as d x dim matrix

         // Compute trace(Du^T Du) = sum_{i,j} (âˆ‚_j u_i)^2
         double trace = 0.0;
         for (int i = 0; i < grad.Height(); i++)
            for (int j = 0; j < grad.Width(); j++)
               trace += grad(i, j) * grad(i, j);

         return trace;
      }
  };

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.UniformRefinement();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // 4. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 5. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 6. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   
   const int max_iter = 20;
   const double tol = 1e-8;

   GridFunction x_new(&fespace);
   Vector residual, dx, B, X;
   SparseMatrix A;

   for (int it = 0; it < max_iter; it++)
   {
      std::cout << "Newton iteration " << it << std::endl;

      // Step 3.1: Build nonlinear coefficient from current x
      GradientSquaredTraceCoefficient a_coeff(x);

      // Step 3.2: Build nonlinear bilinear form
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator);
      a.AddDomainIntegrator(new MassIntegrator(a_coeff));
      a.Assemble();

      // Step 3.3: Compute residual: r = b - A(x)*x
      a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

      residual.SetSize(B.Size());
      A.Mult(X, residual);
      residual *= -1.0;       // residual = -A * X
      residual += B;          // residual = B - A * X

      double res_norm = residual.Norml2();
      std::cout << "  Residual norm = " << res_norm << std::endl;
      if (res_norm < tol)
         break;

      // Step 3.4: Solve linearized system A dx = r
      GSSmoother M(A);
      dx.SetSize(X.Size());
      PCG(A, M, residual, dx, 1, 200, 1e-12, 0.0);

      // Step 3.5: Update solution: X += dx
      X += dx;

      // Step 3.6: Convert X back to GridFunction x
      a.RecoverFEMSolution(X, b, x);
   }
   
   socketstream sol_sock("localhost", 19916);
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << x << flush;
   
   // 10. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   x.Save("sol.gf");
   mesh.Save("mesh.mesh");


   return 0;
}