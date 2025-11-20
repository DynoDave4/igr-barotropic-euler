// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_LAGHOS_SOLVER_HPP
#define MFEM_LAGHOS_SOLVER_HPP

#include "mfem.hpp"
#include "laghos_assembly.hpp"
#include "laghos_solver.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace hydrodynamics
{	
	
	
	

class LagrangianIGRHydroOperator : public LagrangianHydroOperator
{
protected:
	mutable ParFiniteElementSpace fespace;
	double alpha = 0.1;
	bool useIGR = true;


public:
    // Constructor: call base constructor first, then initialize your new member
    LagrangianIGRHydroOperator(const int size,
                           ParFiniteElementSpace &h1_fes,
                           ParFiniteElementSpace &h1_fescalar,
                           ParFiniteElementSpace &l2_fes,
                           const Array<int> &ess_tdofs,
                           Coefficient &rho0_coeff,
                           ParGridFunction &rho0_gf,
                           ParGridFunction &gamma_gf,
                           const int source,
                           const double cfl,
                           const bool visc, const bool vort, const bool pa,
                           const double cgt, const int cgiter, double ftz_tol,
                           const int order_q, bool useIGR_)
        : LagrangianHydroOperator(size, h1_fes, l2_fes, ess_tdofs, rho0_coeff, rho0_gf, gamma_gf,
                           source, cfl, visc, vort, pa,
                           cgt, cgiter, ftz_tol, order_q), 
						   fespace(h1_fescalar), useIGR(useIGR_) {}

   void UpdateQuadratureDataIGR(const Vector &S) const;
   void UpdateQuadratureData(const Vector &S) const override{
	   if(useIGR){	   UpdateQuadratureDataIGR(S); }
	   else {LagrangianHydroOperator::UpdateQuadratureData(S);}
   };
	
   void CalcIGRTerm(ParGridFunction &Phi, ParGridFunction &PhiDot, ParGridFunction &x) const;
   
   void SetAlpha(double a){
	   alpha = a;
   }

};







} // namespace hydrodynamicsIGR

} // namespace mfemIGR

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS
