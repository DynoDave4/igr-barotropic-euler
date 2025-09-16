
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class MassMatrix1 : public MatrixCoefficient
{
   private:
      GridFunction &phi; // vector-valued GridFunction
   public:
      MassMatrix1(GridFunction &phi_) : MatrixCoefficient(phi_.FESpace()->GetVDim()), phi(phi_) {}

   virtual void Eval(DenseMatrix &M, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
	  int dim = phi.FESpace()->GetVDim();

      DenseMatrix Jac;
      phi.GetVectorGradient(T, Jac);  // Jac(i,j) = d phi_i / dx_j

	  DenseMatrix Jac_inv, Jac_invT;
	  Jac_inv = Jac;
	  Jac_inv.Invert();
	  
	  Jac_invT = Jac_inv;
	  Jac_invT.Transpose();
	  
	  M.SetSize(dim, dim);
	  M = 0.0;
	  AddMult(Jac_inv, Jac_invT, M);
   }
};

class Lambda1 : public Coefficient
{
   private:
      GridFunction &phi; // vector-valued GridFunction
   public:
      Lambda1(GridFunction &phi_) : Coefficient(), phi(phi_) {}

   virtual double Eval(ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      
	  
	  //Vector grad;
      //u->GetGradient(T, ip, grad); // compute ∇u at this quadrature point
      return 0.0; 
   }
};

class DivergenceCoefficient : public Coefficient
{
   private:
      VectorCoefficient &vcoeff;  // the input vector field

   public:
      DivergenceCoefficient(VectorCoefficient &vc) : vcoeff(vc) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
		T.SetIntPoint(&ip);
        int dim = vcoeff.GetVDim();
        //Vector val(dim);
        Vector grad(dim*dim);  // flattened gradient matrix

        // Evaluate vector and its gradient
        //vcoeff.Eval(val, T);                // value (not really needed here)
        vcoeff.EvalGradient(grad, T);       // fills flattened Jacobian

        // Compute divergence = sum_i d v_i / dx_i
        double div = 0.0;
        for (int i = 0; i < dim; i++)
        {
            div += grad[i*dim + i];  // assuming column-major flatten
        }
        return div;
    }
};




int main(int argc, char *argv[])
{


return 0;

}



