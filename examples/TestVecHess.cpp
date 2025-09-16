#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


class TestVecHess : public VectorCoefficient
{
   private:
      GridFunction &phi; // vector-valued GridFunction
   public:
      TestVecHess(GridFunction &phi_) : VectorCoefficient(phi_.FESpace()->GetVDim()), phi(phi_) {}

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
	  int dim = 2;

	  const int elem = T.ElementNo;

      // Single-point integration rule -> ?? idk but ChatGPT said to do this to make ip an ir
      IntegrationRule ir;
      ir.SetSize(1);
      ir[0] = ip;

      // Hessians of ALL components stacked
      DenseMatrix all_hess(phi.FESpace()->GetVDim() * dim, dim);
      phi.GetHessians(elem, ir, all_hess, phi.FESpace()->GetVDim());
	    
	  
	  V.SetSize(dim);    
	  V(0) = all_hess(0,0);
	  V(1) = all_hess(2,0);
	  
   }
};


int main(int argc, char *argv[])
{

   //Mesh
   //const char *mesh_file = "../data/star.mesh";
   const char *mesh_file = "../data/periodic-square.mesh";
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   mesh.UniformRefinement();
   mesh.UniformRefinement();
   
   
   //Finite Element Space
   FiniteElementCollection *fec;
   int order = 2;
   fec = new H1_FECollection(order, dim);
   
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;
   FiniteElementSpace feVECspace(&mesh, fec, dim);

   
   
   //Here's the test
   VectorFunctionCoefficient identity(mesh.Dimension(),
    [](const Vector &x, Vector &y) { y = x; });
   GridFunction Phi(&feVECspace);
   Phi.ProjectCoefficient(identity);
   TestVecHess Lpt1(Phi);
   
   //The previous example experiences an error when we add the bilinear form integrator
   BilinearForm a(&fespace);
   InnerProductCoefficient Lambda1(Lpt1, Lpt1);             //This is a scalar coefficient
   a.AddDomainIntegrator(new MassIntegrator(Lambda1));
   a.Assemble();
   

   
   delete fec;
   return 1;

}