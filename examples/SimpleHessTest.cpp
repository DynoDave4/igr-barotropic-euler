#include "mfem.hpp"
#include <iostream>
using namespace mfem;
using namespace std;

// Define a simple quadratic function
double u_exact(const Vector &x)
{
   return x(0)*x(0) + x(1)*x(1);
}

int main(int argc, char *argv[])
{
   // 1. Initialize MFEM and build a mesh
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);

   // 2. Define a quadratic H1 space (order >= 2 is needed for Hessians)
   int order = 2;
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);

   // 3. Project the known function into a GridFunction
   GridFunction u(&fespace);
   FunctionCoefficient u_coeff(u_exact);
   u.ProjectCoefficient(u_coeff);

   // 4. Loop over elements, compute Hessians at quadrature points
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const FiniteElement &fe = *fespace.GetFE(e);
      ElementTransformation &T = *fespace.GetElementTransformation(e);

      // Choose an integration rule of sufficient order
      const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(), 2*order);

      DenseMatrix hess; // will hold Hessian at each ip
      cout << "Element " << e << ":\n";
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         T.SetIntPoint(&ip);

         // Compute Hessian
         //u.GetHessians(e, ip, hess);

         //cout << " ip " << i
         //     << " at (" << ip.x << ", " << ip.y << ")"
         //     << " Hessian = \n";
         //hess.Print(cout);
      }
   }

   return 0;
}
