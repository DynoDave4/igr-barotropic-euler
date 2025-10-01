//                                MFEM Example 1
//
// Compile with: make ex1
//
//               ex1 -m ../data/inline-quad.mesh
//


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

//
// Solve -Δu = 5π^2 cos(2πx)cos(πy) 
// with Neumann BC du/dn = 0 on the domain of Ω = [0, 1]x[0, 1]
// 
// Its exact solution is u = cos(2πx)cos(πy)
// 
// Note that there is a compatibiliy condition that requires int_Ω rhs = 0. 
//
double rhs_func(const Vector &x)
{
   return 5.0*M_PI*M_PI*cos(2.0*M_PI*x(0)) * cos(M_PI*x(1));
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   {
      int ref_levels =
         (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

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

   //This is not needed any more
   Array<int> ess_tdof_list;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm b(&fespace);
   FunctionCoefficient rhs_coeff(rhs_func);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   b.Assemble();


   ConstantCoefficient one(1.0);
   BilinearForm a(&fespace);
   a.SetAssemblyLevel(AssemblyLevel::FULL);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();
   a.Finalize();

   SparseMatrix &A = a.SpMat();
   Vector &B = b;
   Vector X(A.Height());
   X=0.0;

   cout << "Size of linear system: " << A.Height() << endl;

   GSSmoother M(A);
   CGSolver pcg;
   OrthoSolver OrthoPC;
   OrthoPC.SetSolver(M);

   pcg.SetPrintLevel(1);
   pcg.SetMaxIter(200);
   pcg.SetRelTol(1e-6);
   pcg.SetAbsTol(sqrt(0.0));
   pcg.SetOperator(A);
   pcg.SetPreconditioner(OrthoPC);
   pcg.Mult(B, X);

   GridFunction x(&fespace);
   x.MakeRef(&fespace, X, 0);

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
