#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


class BlankDouble : public Coefficient
{
   private:
      GridFunction &phi; // vector-valued GridFunction
	  GridFunction &phidot; // vector-valued GridFunction
   public:
      BlankDouble(GridFunction &phi_, GridFunction &phidot_) : phi(phi_), phidot(phidot_) {}

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {

      return 1;
   }
};

class MyFirstMatrixCoefficient : public MatrixCoefficient
{
private:
   DenseMatrix &A; // could be computed from other data

public:
   MyFirstMatrixCoefficient(int dim, DenseMatrix &A_) 
      : MatrixCoefficient(dim), A(A_) { }

   virtual void Eval(DenseMatrix &M,
                     ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
       // Example: just copy the stored dense matrix
       M = A;
   }
};

class MySecondMatrixCoefficient : public MatrixCoefficient
{
private:
    GridFunction &Phi;  // vector field

public:
    MySecondMatrixCoefficient(int dim, GridFunction &Phi_)
        : MatrixCoefficient(dim), Phi(Phi_) { }

    virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                      const IntegrationPoint &ip)
    {
        Vector grad1, grad2, phi_val;
		
		T.SetIntPoint(&ip);
        //Phi.GetVectorValue(T, phi_val);  // or GetValue for scalar components

        // Fill M based on grad1, grad2, phi_val
        //M.SetSize(grad1.Size(), grad2.Size());
        // Example: outer product of grad1 + phi_val with grad2
        //for (int i = 0; i < grad1.Size(); i++)
        //    for (int j = 0; j < grad2.Size(); j++)
        //        M(i,j) = grad1(i) + phi_val(i) * grad2(j);
    }
};


//Should be working
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

class LambdaDivPart : public VectorCoefficient
{
   private:
      GridFunction &phi; // vector-valued GridFunction
   public:
      LambdaDivPart(GridFunction &phi_) : VectorCoefficient(phi_.FESpace()->GetVDim()), phi(phi_) {}

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.SetIntPoint(&ip);
	  int dim = phi.FESpace()->GetVDim();

      DenseMatrix Jac;
      phi.GetVectorGradient(T, Jac);  // Jac(i,j) = d phi_i / dx_j


	  const int elem = T.ElementNo;

      // Single-point integration rule -> ?? idk but ChatGPT said to do this to make ip an ir
      IntegrationRule ir;
      ir.SetSize(1);
      ir[0] = ip;

      // Hessians of ALL components stacked
      DenseMatrix all_hess(phi.FESpace()->GetVDim() * dim, dim);
      phi.GetHessians(elem, ir, all_hess, phi.FESpace()->GetVDim());
	  
	  // DenseMatrix hessX(dim, dim), hessY(dim, dim);
	  DenseMatrix Ax(dim, dim), Ay(dim, dim);
	  for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
			//These aren't needed, but I'll keep it here for a reference
            //hessX(i,j) = all_hess(i,j); // first 'dim' rows
			//hessY(i,j) = all_hess(i+dim,j);
			Ax(i,j) = all_hess(i*dim,j);       // Ax = ∂x[DΦ]
			Ay(i,j) = all_hess(i*dim+1,j);     // Ay = ∂y[DΦ]
			
         }
      }
	  
	  DenseMatrix Jac_inv;
	  Jac_inv = Jac;
	  Jac_inv.Invert();
	  
	  DenseMatrix Bx(dim, dim), By(dim, dim);
	  Bx = 0.0; AddMult(Ax, Jac_inv, Bx);
	  By = 0.0; AddMult(Ay, Jac_inv, By);
	  
	  DenseMatrix Cx(dim, dim), Cy(dim, dim);
      Cx = 0.0; AddMult(Jac_inv, Bx, Cx);  // Cx = [DΦ]^{-1} ∂x[DΦ] [DΦ]^{-1}
	  Cy = 0.0; AddMult(Jac_inv, By, Cy);  // Cy = [DΦ]^{-1} ∂y[DΦ] [DΦ]^{-1}
	  
	  V.SetSize(dim);    //Coefficients of Lambda2
	  V(0) = -1*Cx(0,0) - Cy(1,0);
	  V(1) = -1*Cx(0,1) - Cy(1,1);
	  
	  
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
	  
	  M.SetSize(dim, dim);
	  M = Jac;
	  M.Invert();
   }
};

class RHSg : public Coefficient
{
   private:
	  GridFunction &phidot; // vector-valued GridFunction
   public:
      RHSg(GridFunction &phidot_) : phidot(phidot_) {}

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
	  T.SetIntPoint(&ip);
	  int dim = phidot.FESpace()->GetVDim();

      DenseMatrix Jac(dim, dim), JacSq(dim, dim);
      phidot.GetVectorGradient(T, Jac);
	  JacSq = 0.0;
	  AddMult(Jac, Jac, JacSq);
	  
      return (Jac(0,0) + Jac(1,1))*(Jac(0,0) + Jac(1,1)) + (JacSq(0,0) + JacSq(1,1));
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
   return 1;

}