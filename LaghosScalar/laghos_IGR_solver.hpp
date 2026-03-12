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
#include "laghos_assembly_IGR.hpp"
#include "laghos_solver.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace hydrodynamics
{


class QUpdateIGR
{
private:
   const int dim, vdim, NQ, NE, Q1D;
   const bool use_vorticity;
   mutable bool use_viscosity;
   const double cfl;
   TimingData *timer;
   const IntegrationRule &ir;
   ParFiniteElementSpace &H1, &H1_scal, &L2;
   const Operator *H1R;
   Vector q_dt_est, q_e, e_vec, q_dx, q_dv, q_igr;
   const QuadratureInterpolator *q1,*q2,*q3;    //q3 is for igr
   const ParGridFunction &gamma_gf;
   double alpha = 0.001;
public:
   QUpdateIGR(const int d, const int ne, const int q1d,
           const bool visc, const bool vort,
           const double cfl, TimingData *t,
           const ParGridFunction &gamma_gf,
           const IntegrationRule &ir,
           ParFiniteElementSpace &h1, ParFiniteElementSpace &h1scal, 
		   ParFiniteElementSpace &l2, double alpha_):
      dim(d), vdim(h1.GetVDim()),
      NQ(ir.GetNPoints()), NE(ne), Q1D(q1d),
      use_viscosity(visc), use_vorticity(vort), cfl(cfl),
      timer(t), ir(ir), H1(h1), H1_scal(h1scal), L2(l2),
      H1R(H1.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
      q_dt_est(NE*NQ),
      q_e(NE*NQ),
	  q_igr(NE*NQ),   //New IGR vector
      e_vec(NQ*NE*vdim),
      q_dx(NQ*NE*vdim*vdim),
      q_dv(NQ*NE*vdim*vdim),
      q1(H1.GetQuadratureInterpolator(ir)),
      q2(L2.GetQuadratureInterpolator(ir)),
	  q3(H1_scal.GetQuadratureInterpolator(ir)),
      gamma_gf(gamma_gf), alpha(alpha_) { }
	  
   void UpdateUseVisc(bool val) { use_viscosity = val;  };

   void UpdateQuadratureData(const Vector &S, QuadratureData &qdata);
   void UpdateQuadratureDataIGR(const Vector &S, QuadratureDataIGR &qdata);
};

// Given a solutions state (x, v, e), this class performs all necessary
// computations to evaluate the new slopes (dx_dt, dv_dt, de_dt).
class LagrangianIGRHydroOperator : public TimeDependentOperator
{
protected:
   //IGR new data
   mutable ParFiniteElementSpace H1_scal;
   bool useIGR = true;
   mutable CGSolver cg_igr;
   mutable HypreBoomerAMG amg_prec; //not used
   double alpha = 0.001;
   double visc_const = 0.0001;
   int visc_type = 3;
   
   //Original Lag
   ParFiniteElementSpace &H1, &L2;
   mutable ParFiniteElementSpace H1c;
   ParMesh *pmesh;
   // FE spaces local and global sizes
   const int H1Vsize;
   const int H1TVSize;
   const HYPRE_BigInt H1GTVSize;
   const int L2Vsize;
   const int L2TVSize;
   const HYPRE_BigInt L2GTVSize;
   Array<int> block_offsets;
   // Reference to the current mesh configuration.
   mutable ParGridFunction x_gf;
   const Array<int> &ess_tdofs;
   const int dim, NE, l2dofs_cnt, h1dofs_cnt, source_type;
   const double cfl;
   const bool use_vorticity, p_assembly;
   mutable bool use_viscosity;
   const double cg_rel_tol;
   const int cg_max_iter;
   const double ftz_tol;
   const ParGridFunction &gamma_gf;
   // Velocity mass matrix and local inverses of the energy mass matrices. These
   // are constant in time, due to the pointwise mass conservation property.
   mutable ParBilinearForm Mv;
   SparseMatrix Mv_spmat_copy;
   DenseTensor Me, Me_inv;
   // Integration rule for all assemblies.
   const IntegrationRule &ir;
   // Data associated with each quadrature point in the mesh.
   // These values are recomputed at each time step.
   const int Q1D;
   mutable QuadratureData qdata;
   //mutable QuadratureDataIGR qdataIGR;
   mutable bool qdata_is_current, forcemat_is_assembled;
   // Force matrix that combines the kinematic and thermodynamic spaces. It is
   // assembled in each time step and then it is used to compute the final
   // right-hand sides for momentum and specific internal energy.
   mutable MixedBilinearForm Force;
   // Same as above, but done through partial assembly.
   ForcePAOperator *ForcePA;
   // Mass matrices done through partial assembly:
   // velocity (coupled H1 assembly) and energy (local L2 assemblies).
   MassPAOperator *VMassPA, *EMassPA;
   OperatorJacobiSmoother *VMassPA_Jprec;
   // Linear solver for energy.
   CGSolver CG_VMass, CG_EMass;
   mutable TimingData timer;
   mutable QUpdateIGR *qupdate;
   mutable Vector X, B, one, rhs, e_rhs;
   mutable ParGridFunction rhs_c_gf, dvc_gf;
   mutable Array<int> c_tdofs[3];

   virtual void ComputeMaterialProperties(int nvalues, const double gamma[],
                                          const double rho[], const double e[],
                                          double p[], double cs[]) const
   {
      for (int v = 0; v < nvalues; v++)
      {
         p[v]  = (gamma[v] - 1.0) * rho[v] * e[v];
         cs[v] = sqrt(gamma[v] * (gamma[v]-1.0) * e[v]);
      }
   }

   void UpdateQuadratureData(const Vector &S) const;
   void AssembleForceMatrix() const;

public:
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
                           const int order_q, bool useIGR_);
   ~LagrangianIGRHydroOperator();
   
   //New IGR Methods
   void UpdateUseIGR(bool val) { useIGR = val;  };
   void UpdateUseVisc(bool val) { 
             if (qupdate) { qupdate->UpdateUseVisc(val); }
             use_viscosity = val;  };
   void SetAlpha(double a){ alpha = a; }
   void SetViscConst(double vc){ visc_const = vc; }
   void SetViscType(int vt){ visc_type = vt; }
   void CalcIGRTerm(ParGridFunction &u, ParGridFunction &x) const;
   
   

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   virtual MemoryClass GetMemoryClass() const
   { return Device::GetMemoryClass(); }

   void SolveVelocity(const Vector &S, Vector &dS_dt) const;
   void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const;
   void UpdateMesh(const Vector &S) const;

   // Calls UpdateQuadratureData to compute the new qdata.dt_estimate.
   double GetTimeStepEstimate(const Vector &S) const;
   void ResetTimeStepEstimate() const;
   void ResetQuadratureData() const { qdata_is_current = false; }

   // The density values, which are stored only at some quadrature points,
   // are projected as a ParGridFunction.
   void ComputeDensity(ParGridFunction &rho) const;
   double InternalEnergy(const ParGridFunction &e) const;
   double KineticEnergy(const ParGridFunction &v) const;

   int GetH1VSize() const { return H1.GetVSize(); }
   const Array<int> &GetBlockOffsets() const { return block_offsets; }

   void PrintTimingData(bool IamRoot, int steps, const bool fom) const;
};



} // namespace hydrodynamics


} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_LAGHOS
