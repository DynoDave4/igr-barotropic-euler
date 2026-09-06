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
//
//                     __                __
//                    / /   ____  ____  / /_  ____  _____
//                   / /   / __ `/ __ `/ __ \/ __ \/ ___/
//                  / /___/ /_/ / /_/ / / / / /_/ (__  )
//                 /_____/\__,_/\__, /_/ /_/\____/____/
//                             /____/
//
//             High-order Lagrangian Hydrodynamics Miniapp
//
// Laghos(LAGrangian High-Order Solver) is a miniapp that solves the
// time-dependent Euler equation of compressible gas dynamics in a moving
// Lagrangian frame using unstructured high-order finite element spatial
// discretization and explicit high-order time-stepping. Laghos is based on the
// numerical algorithm described in the following article:
//
//    V. Dobrev, Tz. Kolev and R. Rieben, "High-order curvilinear finite element
//    methods for Lagrangian hydrodynamics", SIAM Journal on Scientific
//    Computing, (34) 2012, pp. B606–B641, https://doi.org/10.1137/120864672.
//
// Test problems:
//    p = 0  --> Taylor-Green vortex (smooth problem).
//    p = 1  --> Sedov blast.
//    p = 2  --> 1D Sod shock tube.
//    p = 3  --> Triple point.
//    p = 4  --> Gresho vortex (smooth problem).
//    p = 5  --> 2D Riemann problem, config. 12 of doi.org/10.1002/num.10025
//    p = 6  --> 2D Riemann problem, config.  6 of doi.org/10.1002/num.10025
//    p = 7  --> 2D Rayleigh-Taylor instability problem.
//    p = 8  --> Linear C0 shock (not differentiable)
//    p = 9  --> Gaussian Blast Wave
//    p = 10 --> Smooth Sod Shock Tube
//    p = 11 --> Mach Test (v0 = A sine(.))
//    p = 12 --> NonSmooth Shock version of p=13
//    p = 13 --> Smooth LeBlanc Tube
//    p = 14 --> Shu Osher
//    p = 15 --> Smooth Triple Point (controlled by -sharp)
//    p = 16 --> Smooth Pressure Triple Point but gamma0, e0, rho0 discontinuous 
//    p = 17 --> 1d Shock hits material
//    p = 18 --> Richtmeyer Meshkov
//    p = 19 --> Case 17 with a monotone mesh-width material transition
//    p = 20 --> Multiple Blasts
//
// Sample runs: see README.md, section 'Verification of Results'.
//

#include <fstream>
#include <sys/time.h>
#include <sys/resource.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <limits>
#include "laghos_IGR_solver.hpp"
#include "fem/qinterp/eval.hpp"
#include "fem/qinterp/det.hpp"
#include "fem/qinterp/grad.hpp"
#include "fem/integ/bilininteg_mass_kernels.hpp"

#include "sedov/sedov_sol.hpp"
#ifdef LAGHOS_USE_CALIPER
#include <caliper/cali.h>
#include <adiak.hpp>
#endif

#if (defined(HYPRE_USING_UMPIRE) || defined(MFEM_USE_UMPIRE)) && (defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))
#define LAGHOS_USE_DEVICE_UMPIRE
#define LAGHOS_DEVICE_ALLOCATOR_NAME "LAGHOS_DEVICE_POOL"
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>

double getDeviceMemoryHighWatermark() {
  auto &rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator(LAGHOS_DEVICE_ALLOCATOR_NAME);
  return ((double) allocator.getHighWatermark()) / (1024 * 1024 * 1024);
}
#endif

using std::cout;
using std::endl;
using namespace mfem;

// Choice for the problem setup.
static int problem, dim;
real_t Sx = 1, Sy = 1, Sz = 1;  // Sx was "length" in my previous code
int sharpness = 10;
int material_transition_points = 4;
double material_mesh_h[3] = {1.0, 1.0, 1.0};

// Forward declarations.
double e0(const Vector &);
double rho0(const Vector &);
double gamma_func(const Vector &);
void v0(const Vector &, Vector &);

static double Case19Transition(double x, double x0, double h);
static double Case19Density(const Vector &x);
static double Case19Gamma(const Vector &x);
static void Case19EnergyBounds(double &emin, double &emax);
static void BoundGridFunction(ParGridFunction &gf, double min_val, double max_val);
static void SetMaterialMeshSpacing(const ParMesh &pmesh);
static void AssignMeshBdrAttrs2D(Mesh &, real_t, real_t);
static void AssignMeshBdrAttrs3D(Mesh &, real_t, real_t, real_t, real_t);

static long GetMaxRssMB();
static void display_banner(std::ostream&);
static void Checks(const int ti, const double norm, int &checks);

double smooth(const Vector &x) {
	     return exp(-100*(pow(x[0]-0.5,2) + pow(x[1]-0.5,2)));
      };
	  
double Gauss(double x) {return exp(-1*x*x); };
	  
class Gaussian : public mfem::Coefficient
{
private:
   double var, h, cx, cy, cz;

public:
   Gaussian(double var_, double h_, double cx_, double cy_, double cz_)
      : var(var_), h(h_), cx(cx_), cy(cy_), cz(cz_) {}

   virtual double Eval(mfem::ElementTransformation &T,
                       const mfem::IntegrationPoint &ip)
   {
      mfem::Vector x;
      T.Transform(ip, x);   // x = physical coordinates
	  int dim = T.GetSpaceDim();
	  double out = h;
     double center[3] = {cx, cy, cz};
	  for(int i=0; i<dim; i++){
		  out *= Gauss((x[i] - center[i]) / sqrt(2*var)) / sqrt(2*3.141592*var);
	  }
      return out;
   }
};


#ifdef LAGHOS_USE_CALIPER
   static void RecordAdiakMetadata(int dim, const char *mesh_file, int elem_per_mpi,
                                 int nx, int ny, int nz, double blast_energy,
                                 double Sx, double Sy, double Sz,
                                 int rs_levels, int rp_levels, int problem,
                                 int order_v, int order_e, int order_q,
                                 int ode_solver_type, double t_final, double cfl,
                                 double cg_tol, double ftz_tol, double delta_tol,
                                 int cg_max_iter, int max_tsteps,
                                 bool p_assembly, bool impose_visc,
                                 bool visualization, int vis_steps, bool visit,
                                 bool gfprint, const char *basename,
                                 const char *device, bool check,
                                 bool check_exact_sedov, bool mem_usage,
                                 bool fom, bool gpu_aware_mpi, int dev_pool_size,
                                 bool enable_nc, int dev);
#endif

int main(int argc, char *argv[])
{

   //Start Timer
   auto start = std::chrono::high_resolution_clock::now();

   // Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Print the banner.
   if (Mpi::Root()) { display_banner(cout); }

   // Parse command-line options.
   problem = 1;
   dim = 3;
   const char *mesh_file = "default";
   int elem_per_mpi = 0;
   int rs_levels = 2;
   int rp_levels = 0;
   int nx = 2;
   int ny = 2;
   int nz = 2;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   int ode_solver_type = 3;
   double t_final = 0.6;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   double ftz_tol = 0.0;
   double delta_tol = 1e-12;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool p_assembly = false;
   bool impose_visc = false;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   const char *basename = "results/Laghos";
   const char *device = "cpu";
   bool check = false;
   bool check_exact_sedov = false;
   bool mem_usage = false;
   bool fom = false;
   bool gpu_aware_mpi = false;
   int dev = 0;
   int dev_pool_size = 4;

   double blast_energy = 1;

   bool enable_nc = true;

   //New IGR variables
   double alpha = 4.0;
   double C_epsilon = 8.0*alpha;
   double stallIGR = -0.3;
   double e_reg = 0.0;
   bool useIGR = true;
   bool corner = false;    // Used for position Sedov and Gaussian blasts
   double variance = 0.1;
   double visc_const = 250.1234;  // Viscosity constant i.e. A
   int visc_type = 3;  // 1 is Laghos Artificial Visc, 2 is const A, 3 is dx(A ||u|| + c)
   int alpha_type = 3; // 1 is const alpha, 2 is function, 3 is const*dx^2, 4 - min dx, 5 - max dx
                       // 6 uses J^T J, 7 uses J J^T   ---> NOT implemented here 
   bool TestPrint = false;
   bool gfread = false;
   const char *ParaPre = "ParaView/";
   bool paraview = false;
   bool parabolic = false;
   int frames = 100;
   double dtmax = -1;

   #ifdef LAGHOS_USE_CALIPER
      cali_config_set("CALI_CALIPER_ATTRIBUTE_DEFAULT_SCOPE", "process");
      CALI_CXX_MARK_FUNCTION;
   
      MPI_Comm adiak_mpi_comm = MPI_COMM_WORLD;
      void* adiak_mpi_comm_ptr = &adiak_mpi_comm;
      adiak::init(adiak_mpi_comm_ptr);
      adiak::collect_all();
   #endif

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(
      &elem_per_mpi, "-epm", "--elem-per-mpi",
      "Number of element per mpi task. Note: this is mutually-exclusive with "
      "-nx, -ny, and -nz. Use -epm 0 to use -nx, -ny, and -nz.");
   args.AddOption(&nx, "-nx", "--xelems",
                  "Elements in x-dimension (do not specify mesh_file). Note: "
                  "this is mutually-exclusive with -epm. Use -epm "
                  "0 to use -nx, -ny, and -nz.");
   args.AddOption(&ny, "-ny", "--yelems",
                  "Elements in y-dimension (do not specify mesh_file). Note: "
                  "this is mutually-exclusive with -epm. Use -epm "
                  "0 to use -nx, -ny, and -nz.");
   args.AddOption(&nz, "-nz", "--zelems",
                  "Elements in z-dimension (do not specify mesh_file). Note: "
                  "this is mutually-exclusive with -epm. Use -epm "
                  "0 to use -nx, -ny, and -nz.");
   args.AddOption(&blast_energy, "-E0", "--blast-energy",
                  "Sedov initial blast energy (for problem 1)");
   args.AddOption(&Sx, "-Sx", "--xwidth",
                  "Domain width in x-dimension (do not specify mesh_file)");
   args.AddOption(&Sy, "-Sy", "--ywidth",
                  "Domain width in y-dimension (do not specify mesh_file)");
   args.AddOption(&Sz, "-Sz", "--zwidth",
                  "Domain width in z-dimension (do not specify mesh_file)");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_q, "-oq", "--order-intrule",
                  "Order  of the integration rule.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            7 - RK2Avg.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&ftz_tol, "-ftz", "--ftz-tol",
                  "Absolute flush-to-zero tolerance.");
   args.AddOption(&delta_tol, "-dtol", "--delta-tol",
                  "Tolerance for projecting Delta functions.");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&p_assembly, "-pa", "--partial-assembly", "-fa",
                  "--full-assembly",
                  "Activate 1D tensor-based assembly (partial assembly).");
   args.AddOption(&impose_visc, "-iv", "--impose-viscosity", "-niv",
                  "--no-impose-viscosity",
                  "Use active viscosity terms even for smooth problems.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&device, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&check, "-chk", "--checks", "-no-chk", "--no-checks",
                  "Enable 2D checks.");
   args.AddOption(&check_exact_sedov, "-err", "--exact-error", "-no-err",
                  "--no-exact-error",
                  "Enable comparing the Sedov problem (problem 1) against the "
                  "exact solution.");
   args.AddOption(&mem_usage, "-mb", "--mem", "-no-mem", "--no-mem",
                  "Enable memory usage.");
   args.AddOption(&fom, "-f", "--fom", "-no-fom", "--no-fom",
                  "Enable figure of merit output.");
   args.AddOption(&gpu_aware_mpi, "-gam", "--gpu-aware-mpi", "-no-gam",
                  "--no-gpu-aware-mpi", "Enable GPU aware MPI communications.");
   args.AddOption(&dev_pool_size, "-pool", "--dev-pool-size",
                  "Size (in GB) for the umpire device pool");
   args.AddOption(&enable_nc, "-nc", "--nonconforming", "-no-nc",
                  "--conforming",
                  "Use non-conforming meshes. Requires a 2D or 3D mesh.");
   args.AddOption(&dev, "-dev", "--dev", "GPU device to use.");

   // New IGR Flags
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Alpha as the level of IGR");
   args.AddOption(&C_epsilon, "-epsR", "--epsilon-ratio",
                  "Epsilon as the level of parabolic calculation");
   args.AddOption(&corner, "-corner", "--corner-blast", "-center",
                  "--center-blast", "Where does the shockwave start?");
   args.AddOption(&variance, "-var", "--variance",
                  "Variance of Gaussian shockwave");
   args.AddOption(&useIGR, "-igr", "--use-igr", "-noigr",
                  "--no-igr", "Do we add the igr term?");
   args.AddOption(&TestPrint, "-tp", "--test-print", "-notp",
                  "--no-test-print", "Do we !print to a folder?");
   args.AddOption(&stallIGR, "-sigr", "--stall-igr",
                  "Do we run without igr for a bit first?");
   args.AddOption(&e_reg, "-er", "--energy-reg",
                  "Do we set background energy to something?");
   args.AddOption(&sharpness, "-sharp", "--sharpness",
                  "Sets the sharpness of the initial condition.");
   args.AddOption(&material_transition_points, "-mpts",
                  "--material-transition-points",
                  "Number of mesh spacings used to spread the material "
                  "discontinuity in problem 19.");
   args.AddOption(&visc_const, "-vc", "--visc-const",
                  "Sets the viscosity constant.");
   args.AddOption(&visc_type, "-vt", "--visc-type",
                  "Sets the way we add viscosity.");
   args.AddOption(&alpha_type, "-at", "--alpha-type",
                  "Sets the way we add alpha.");
   args.AddOption(&gfread, "-read", "--read", "-no-read", "--no-read",
                  "Enable or disable reading the grid function.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&parabolic, "-parabolic", "--parabolic-igr-calc", "-no-parabolic",
                  "--no-parabolic-igr-calc",
                  "Do we calculate IGR pressure with a conservation law?");
   args.AddOption(&frames, "-frames", "--frames",
                  "Number of ParaView frames.");
   args.AddOption(&dtmax, "-dtmax", "--dt-max",
                  "Sets max dt width if wanted.");   
   bool C_epsilon_arg = false;
   for (int i = 1; i < argc; i++)
   {
      if (strcmp(argv[i], "-epsR") == 0 ||
          strcmp(argv[i], "--epsilon-ratio") == 0)
      {
         C_epsilon_arg = true;
      }
   }
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (!C_epsilon_arg) { C_epsilon = 8.0*alpha; }
   const bool requested_parabolic = parabolic;
   MFEM_VERIFY(material_transition_points > 0,
               "The material transition width must be at least one mesh "
               "spacing.");

   if (Mpi::Root())
   {
      args.PrintOptions(cout);

      #ifdef LAGHOS_USE_CALIPER
         RecordAdiakMetadata(dim, mesh_file, elem_per_mpi, nx, ny, nz,
                          blast_energy, Sx, Sy, Sz, rs_levels, rp_levels,
                          problem, order_v, order_e, order_q, ode_solver_type,
                          t_final, cfl, cg_tol, ftz_tol, delta_tol,
                          cg_max_iter, max_tsteps, p_assembly, impose_visc,
                          visualization, vis_steps, visit, gfprint, basename,
                          device, check, check_exact_sedov, mem_usage, fom,
                          gpu_aware_mpi, dev_pool_size, enable_nc, dev);
      #endif
   }

   if (check_exact_sedov)
   {
      MFEM_VERIFY(
         problem == 1,
         "Can only compare problem 1 (Sedov) against the exact solution");
      MFEM_VERIFY(strncmp(mesh_file, "default", 7) == 0, "check: mesh_file");
   }

#ifdef LAGHOS_USE_DEVICE_UMPIRE
   auto &rm = umpire::ResourceManager::getInstance();
   const char * allocator_name = LAGHOS_DEVICE_ALLOCATOR_NAME;
   size_t umpire_dev_pool_size = ((size_t) dev_pool_size) * 1024 * 1024 * 1024;
   size_t umpire_dev_block_size = 512;
   rm.makeAllocator<umpire::strategy::QuickPool>(allocator_name,
                                                 rm.getAllocator("DEVICE"),
                                                 umpire_dev_pool_size,
                                                 umpire_dev_block_size);

#ifdef HYPRE_USING_UMPIRE
   HYPRE_SetUmpireDevicePoolName(allocator_name);
#endif // HYPRE_USING_UMPIRE

#ifdef MFEM_USE_UMPIRE
   MemoryManager::SetUmpireDeviceAllocatorName(allocator_name);
   // the umpire host memory type is slow compared to the native host memory type
   Device::SetMemoryTypes(MemoryType::HOST, MemoryType::DEVICE_UMPIRE);
#endif // MFEM_USING_UMPIRE
#endif // LAGHOS_USE_DEVICE_UMPIRE

   // Configure the device from the command line options
   Device backend;
   backend.Configure(device, dev);
   if (Mpi::Root()) { backend.Print(); }
   backend.SetGPUAwareMPI(gpu_aware_mpi);

   // Prepare the missing kernels.
   if (myid == 0) { KernelReporter::Enable(); }
   using TENS = QuadratureInterpolator::TensorEvalKernels;
   using DET  = QuadratureInterpolator::DetKernels;
   using GRAD = QuadratureInterpolator::GradKernels;
   // 2D Q1Q0.
   TENS::Specialization<2,QVectorLayout::byNODES,1,1,2>::Opt<1>::Add();
   TENS::Specialization<2,QVectorLayout::byVDIM,1,1,2>::Opt<1>::Add();
   TENS::Specialization<2,QVectorLayout::byVDIM,2,2,2>::Opt<1>::Add();
   GRAD::Specialization<2,QVectorLayout::byVDIM,0,2,2,2>::Add();
   // 2D Q2Q1 - ok.
   // 2D Q3Q2 - ok.
   // 2D Q4Q3.
   TENS::Specialization<2,QVectorLayout::byNODES,1,4,8>::Opt<1>::Add();
   TENS::Specialization<2,QVectorLayout::byVDIM,2,5,8>::Opt<1>::Add();
   DET::Specialization<2,2,5,8>::Add();
   GRAD::Specialization<2,QVectorLayout::byNODES,0,2,5,8>::Add();
   MassIntegrator::AddSpecialization<2,4,8>();
   MassIntegrator::AddSpecialization<2,5,8>();
   // 3D Q1Q0.
   TENS::Specialization<3,QVectorLayout::byNODES,1,1,2>::Opt<1>::Add();
   TENS::Specialization<3,QVectorLayout::byVDIM,1,1,2>::Opt<1>::Add();
   DET::Specialization<3,3,2,2>::Add();
   GRAD::Specialization<3,QVectorLayout::byNODES,0,3,2,2>::Add();
   GRAD::Specialization<3,QVectorLayout::byVDIM,0,3,2,2>::Add();
   // 3D Q2Q1 - ok.
   // 3D Q3Q2 - ok.
   // 3D Q4Q3.
   TENS::Specialization<3,QVectorLayout::byVDIM,3,5,8>::Opt<1>::Add();
   // DET::Specialization<3,3,5,8>::Add(); // not enough shared memory.
   GRAD::Specialization<3,QVectorLayout::byNODES,0,3,5,8>::Add();
   MassIntegrator::AddSpecialization<3,4,8>();

   // On all processors, use the default builtin 1D/2D/3D mesh or read the
   // serial one given on the command line.
   Mesh mesh;
   Array<int> mpi_partitioning;
   if (strncmp(mesh_file, "default", 7) != 0)
   {
#ifndef MFEM_USE_METIS
      MFEM_ABORT("MFEM has not been built with METIS. Use the \"default\" mesh.");
#endif

      // Read the serial mesh from the given mesh file on all processors.
      // Refine the mesh in serial to increase the resolution.
      mesh = Mesh::LoadFromFile(mesh_file, 1, 1);
      for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   }
   else
   {
      if (elem_per_mpi)
      {
         mesh = PartitionMPI(dim, Mpi::WorldSize(), elem_per_mpi, myid == 0,
                             rp_levels, mpi_partitioning);
         // scale mesh by Sx, Sy, Sz
         switch (dim)
         {
            case 1:
               mesh.Transform([=](const Vector &x, Vector &y) { y[0] = x[0] * Sx; });
               mesh.GetBdrElement(0)->SetAttribute(1);
               mesh.GetBdrElement(1)->SetAttribute(1);
               break;
            case 2:
               mesh.Transform([=](const Vector &x, Vector &y)
               {
                  y[0] = x[0] * Sx;
                  y[1] = x[1] * Sy;
               });
               AssignMeshBdrAttrs2D(mesh, 0_r, Sx);
               break;
            case 3:
               mesh.Transform([=](const Vector &x, Vector &y)
               {
                  y[0] = x[0] * Sx;
                  y[1] = x[1] * Sy;
                  y[2] = x[2] * Sz;
               });
               AssignMeshBdrAttrs3D(mesh, 0_r, Sx, 0_r, Sy);
               break;
         }
      }
      else
      {
         if (dim == 1)
         {
            mesh = Mesh::MakeCartesian1D(nx, Sx);
            mesh.GetBdrElement(0)->SetAttribute(1);
            mesh.GetBdrElement(1)->SetAttribute(1);

            // Preserve the centered 1D domain used by the previous Scal3/4
            // problem setups when a non-unit width is requested. In
            // particular, p14 with -Sx 2 uses [-0.5, 1.5], so its fixed
            // initial-condition features remain in their historical places.
            if (Sx != 1.0 && problem > 7)
            {
               for (int i = 0; i < mesh.GetNV(); i++)
               {
                  mesh.GetVertex(i)[0] -= (Sx - 1.0) / 2.0;
               }
            }
         }
         if (dim == 2)
         {
            mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, true, Sx,
                                         Sy);
            AssignMeshBdrAttrs2D(mesh, 0_r, Sx);
         }
         if (dim == 3)
         {
            mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, Sx, Sy,
                                         Sz, true);
            AssignMeshBdrAttrs3D(mesh, 0_r, Sx, 0_r, Sy);
         }
         for (int lev = 0; lev < rs_levels; lev++)
         {
            mesh.UniformRefinement();
         }
      }
   }
   dim = mesh.Dimension();

   // 1D vs partial assembly sanity check.
   if (p_assembly && dim == 1)
   {
      p_assembly = false;
      if (Mpi::Root())
      {
         cout << "Laghos does not support PA in 1D. Switching to FA." << endl;
      }
   }

   if (enable_nc && dim > 1)
   {
      if (Mpi::Root())
      {
         cout << "Using non-conforming mesh." << endl;
      }
      mesh.EnsureNCMesh();
   }

   if(dim == 1 && myid == 0){cout << "Serial dx = " << (Sx / (mesh.GetNV()-1)) << std::endl;}
   if(dim == 1 && myid == 0){cout << "Serial dx^2 = " << (Sx / (mesh.GetNV()-1))*(Sx / (mesh.GetNV()-1)) << std::endl;}
   if(dim == 2 && myid == 0){cout << "Serial dx^2 is about " << Sx*Sy / mesh.GetNV() << std::endl;}
   const int mesh_NE = mesh.GetNE();
   if (Mpi::Root())
   {
      cout << "Number of zones in the serial mesh: " << mesh_NE << endl;
   }

   // Parallel partitioning of the mesh.
   // Refine the mesh further in parallel to increase the resolution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh, mpi_partitioning.GetData());
   mesh.Clear();
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }
   SetMaterialMeshSpacing(pmesh);
   if (problem == 19 && myid == 0)
   {
      cout << "Problem 19 material transition width: "
           << material_transition_points << " mesh spacing(s)." << endl;
   }

   int NE = pmesh.GetNE(), ne_min, ne_max;
   MPI_Reduce(&NE, &ne_min, 1, MPI_INT, MPI_MIN, 0, pmesh.GetComm());
   MPI_Reduce(&NE, &ne_max, 1, MPI_INT, MPI_MAX, 0, pmesh.GetComm());
   if (myid == 0)
   { cout << "Zones min/max: " << ne_min << " " << ne_max << endl; }

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(&pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(&pmesh, &H1FEC, pmesh.Dimension());
   ParFiniteElementSpace H1FEScalarSpace(&pmesh, &H1FEC, 1);

   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs, ess_vdofs;
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max()), dofs_marker, dofs_list;
      for (int d = 0; d < pmesh.Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries,
         // i.e., we must enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, d);
         ess_tdofs.Append(dofs_list);
         H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, d);
         FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
         ess_vdofs.Append(dofs_list);
      }
   }

   // Define the explicit ODE solver used for time integration.
   ODESolver *ode_solver = nullptr;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      case 7: ode_solver = new RK2AvgSolver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         MPI_Finalize();
         return 3;
   }

   const HYPRE_BigInt glob_size_l2 = L2FESpace.GlobalTrueVSize();
   const HYPRE_BigInt glob_size_h1 = H1FESpace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of kinematic (position, velocity) dofs: "
           << glob_size_h1 << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_l2 << endl;
   }

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy
   const int Vsize_l2 = L2FESpace.GetVSize();
   const int Vsize_h1 = H1FESpace.GetVSize();
   const int Vsize_igr = H1FEScalarSpace.GetVSize();
   Array<int> offset(5);
   offset[0] = 0;
   offset[1] = offset[0] + Vsize_h1;
   offset[2] = offset[1] + Vsize_h1;
   offset[3] = offset[2] + Vsize_l2;
   offset[4] = offset[3] + Vsize_igr;
   BlockVector S(offset, Device::GetMemoryType());

   // Define GridFunction objects for the position, velocity and specific
   // internal energy. There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf, v_gf, e_gf, igr_gf;
   x_gf.MakeRef(&H1FESpace, S, offset[0]);
   v_gf.MakeRef(&H1FESpace, S, offset[1]);
   e_gf.MakeRef(&L2FESpace, S, offset[2]);
   igr_gf.MakeRef(&H1FEScalarSpace, S, offset[3]);

   // Initialize x_gf using the starting mesh coordinates.
   pmesh.SetNodalGridFunction(&x_gf);
   // Sync the data location of x_gf with its base, S
   x_gf.SyncAliasMemory(S);

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh.Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);
   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      v_gf(ess_vdofs[i]) = 0.0;
   }
   // Sync the data location of v_gf with its base, S
   v_gf.SyncAliasMemory(S);

   // Initialize density and specific internal energy values. We interpolate in
   // a non-positive basis to get the correct values at the dofs. Then we do an
   // L2 projection to the positive basis in which we actually compute. The goal
   // is to get a high-order representation of the initial condition. Note that
   // this density is a temporary function and it will not be updated during the
   // time evolution.
   ParGridFunction rho0_gf(&L2FESpace);
   FunctionCoefficient rho0_coeff(rho0);
   L2_FECollection l2_fec(order_e, dim);
   ParFiniteElementSpace l2_fes(&pmesh, &l2_fec);
   ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes), l2_one(&l2_fes), l2_be2(&l2_fes), l2_be3(&l2_fes);
   l2_rho0_gf.ProjectCoefficient(rho0_coeff);
   rho0_gf.ProjectGridFunction(l2_rho0_gf);
   if (problem == 19) { BoundGridFunction(rho0_gf, 0.2, 1.0); }

   double blast_position[] = {0.0, 0.0, 0.0};
   if(!corner){for(int i=0; i<3; i++){blast_position[i] = 0.5;}}
   if (problem == 1)
   {
      // For the Sedov test, we use a delta function at the origin.
      // divide amount of blast energy by 2^d due to simulating only a portion
      // of the symmetric blast.
      DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                               blast_position[2], blast_energy / pow(2, dim));
      e_coeff.SetTol(delta_tol);
      l2_e.ProjectCoefficient(e_coeff);

      int non_finite = l2_e.CheckFinite();
      MPI_Allreduce(MPI_IN_PLACE, &non_finite, 1, MPI_INT, MPI_SUM, pmesh.GetComm());
      if (non_finite > 0)
      {
         cout << "Delta function could not be initialized!\n";
         delete ode_solver;
         return 1;
      }
   }
   else if(problem == 9){
	  // For the Sedov test, we use a delta function at the origin.
	  ConstantCoefficient reg(e_reg);
	  l2_one.ProjectCoefficient(reg);
	  
	  Gaussian smoothblast(variance, blast_energy, 
              blast_position[0], blast_position[1], blast_position[2]);
      l2_e.ProjectCoefficient(smoothblast);
	  l2_e += l2_one;
   }
   else if(problem == 20){
     ConstantCoefficient reg(e_reg);
	  l2_one.ProjectCoefficient(reg);
	  
	  Gaussian smoothblast1(2*variance, blast_energy*8, 0.35, 0.35, 0.35);
     Gaussian smoothblast2(variance/2, blast_energy/2, 0.425, 0.68, 0.74);
     Gaussian smoothblast3(variance, blast_energy*2, 0.65, 0.5, 0.575);
     
     l2_e.ProjectCoefficient(smoothblast1);
     l2_be2.ProjectCoefficient(smoothblast2);
     l2_be3.ProjectCoefficient(smoothblast3);
	  l2_e += l2_one;
     l2_e += l2_be2;
     l2_e += l2_be3;

   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);
   if (problem == 19)
   {
      double e_min, e_max;
      Case19EnergyBounds(e_min, e_max);
      BoundGridFunction(e_gf, e_min, e_max);
   }
   // Sync the data location of e_gf with its base, S
   e_gf.SyncAliasMemory(S);

   igr_gf = 0.0;
   igr_gf.SyncAliasMemory(S);

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on function that's constant on the moving mesh.
   L2_FECollection mat_fec(0, dim);
   ParFiniteElementSpace mat_fes(&pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma_func);
   mat_gf.ProjectCoefficient(mat_coeff);

   // Additional details, depending on the problem.
   int source = 0; bool visc = false, vorticity = false;
   switch (problem)
   {
      case 0: if (dim == 2) { source = 1; } break;
      case 1: break;
      case 2: break;
      case 3: S.HostRead(); break;
      case 4: break;
      case 5: break;
      case 6: break;
      case 7: source = 2; vorticity = true;  break;
      case 8: break;
	   case 9: break;
	   case 10: break;
      case 11: break;
      case 12: break;
      case 13: break;
      case 14: break;
      case 15: S.HostRead(); break;
      case 16: S.HostRead(); break;
      case 17: break;
      case 18: break;
      case 19: break;
      case 20: break;
      default: MFEM_ABORT("Wrong problem specification!");
   }
   if (impose_visc) { visc = true; }

   hydrodynamics::LagrangianHydroOperator hydro(S.Size(),
                                                H1FESpace, H1FEScalarSpace, L2FESpace, 
                                                ess_tdofs,
                                                rho0_coeff, rho0_gf,
                                                mat_gf, source, cfl,
                                                visc, vorticity, p_assembly,
                                                cg_tol, cg_max_iter, ftz_tol,
                                                order_q, useIGR,
                                                alpha, alpha_type, false,
                                                C_epsilon);
   hydro.SetViscConst(visc_const);
   hydro.SetViscType(visc_type);

   socketstream vis_rho, vis_v, vis_e, vis_igr, vis_p;
   char vishost[] = "localhost";
   int  visport   = 19916;

   ParGridFunction rho_gf;
   const bool plot_pressure = (problem == 3 || problem == 15 || problem == 16);
   ParGridFunction p_gf(&L2FESpace);
   GridFunctionCoefficient rho_gf_coeff(&rho_gf);
   GridFunctionCoefficient e_gf_coeff(&e_gf);
   GridFunctionCoefficient gamma_gf_coeff(&mat_gf);
   ConstantCoefficient minus_one(-1.0);
   SumCoefficient gamma_minus_one(gamma_gf_coeff, minus_one);
   ProductCoefficient rho_e(rho_gf_coeff, e_gf_coeff);
   ProductCoefficient pressure_coeff(gamma_minus_one, rho_e);
   if (visualization || visit) { hydro.ComputeDensity(rho_gf); }
   if (visualization && plot_pressure) { p_gf.ProjectCoefficient(pressure_coeff); }
   const double energy_init = hydro.InternalEnergy(e_gf) +
                              hydro.KineticEnergy(v_gf);

   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh.GetComm());
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      vis_igr.precision(8);
      vis_p.precision(8);
      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets
      if (problem != 0 && problem != 4)
      {
         hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                       "Density", Wx, Wy, Ww, Wh);
      }
      Wx += offx;
      hydrodynamics::VisualizeField(vis_v, vishost, visport, v_gf,
                                    "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                    "Specific Internal Energy", Wx, Wy, Ww, Wh);
      Wx += offx;
      if(useIGR){
            hydrodynamics::VisualizeField(vis_igr, vishost, visport, igr_gf,
                                       "IGR", Wx, Wy, Ww, Wh);
	   Wx += offx;
	   }
      if (plot_pressure)
      {
         hydrodynamics::VisualizeField(vis_p, vishost, visport, p_gf,
                                       "Pressure", Wx, Wy, Ww, Wh);
      }
   }

   // Save data for VisIt visualization.
   VisItDataCollection visit_dc(basename, &pmesh);
   if (visit)
   {
      visit_dc.RegisterField("Density",  &rho_gf);
      visit_dc.RegisterField("Velocity", &v_gf);
      visit_dc.RegisterField("Specific Internal Energy", &e_gf);
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   // Perform time-integration (looping over the time iterations, ti, with a
   // time-step dt). The object oper is of type LagrangianHydroOperator that
   // defines the Mult() method that used by the time integrators.
   ode_solver->Init(hydro);
   hydro.ResetTimeStepEstimate();
   double t = 0.0, dt = hydro.GetTimeStepEstimate(S), t_old;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   long mem=0, mmax=0, msum=0;
   long dmem = 0, dmmax = 0, dmsum = 0;
   int checks = 0;

      const char *igr_suffix = "", *igr_folder = "WithIGR/";
   std::string problem_folder = "p" + std::to_string(problem) + "/", visc_suffix = "", alpha_folder = "";
   if(!useIGR){
      igr_suffix = "_noigr";
      igr_folder = "WithoutIGR/";
   } else if(alpha < 0.001){
      alpha_folder = "alpha=" + std::to_string(alpha*1000000) + "e-6/";
   } else{
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << alpha*1000;
      alpha_folder = "alpha=" + oss.str() + "e-3/";
   }
   if(visc){
   if(visc_const < 0){ 
      visc_suffix = "_LaghosVisc";
   } else{
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << visc_const;
      visc_suffix =  "_" + oss.str();
   }
   }
   if(TestPrint){ igr_folder = ""; problem_folder = ""; }

   std::ostringstream oss;
   oss << std::setprecision(3) << t_final * 1000;
   std::string t_str = oss.str();
   double para_vis_dt = t_final / frames;
   double para_next_vis_time = 0.0;
   int para_vis_cycle = 0;
   const int output_rs_levels = (rs_levels == 0) ? nx : rs_levels;
   std::string folder = std::string(ParaPre) + igr_folder + alpha_folder + "p" + std::to_string(problem);
   const std::string material_width_suffix =
      (problem == 19) ? "_mpts" + std::to_string(material_transition_points) : "";
   std::string run_name = "Laghos_" + std::to_string(problem) + "_" +
                       "dim" + std::to_string(dim) + "_" +
                       std::to_string(output_rs_levels) + "_" +
                       std::to_string(order_v) +
                       std::to_string(order_e) +
                       std::to_string(ode_solver_type) +
                       std::to_string(visc_type) +
                       std::to_string(alpha_type) + "_" +
                       std::to_string(sharpness) + material_width_suffix + "_" +
                       t_str + igr_suffix + visc_suffix;

   
   auto save_paraview_frame = [&](int cycle, double time, bool restart)
   {
      ParaViewDataCollection pvdc(run_name, &pmesh);
      pvdc.SetPrefixPath(folder);
      pvdc.SetLevelsOfDetail(order_v);
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(true);
      pvdc.UseRestartMode(restart);
   
      pvdc.RegisterField("density", &rho_gf);
      pvdc.RegisterField("velocity", &v_gf);
      pvdc.RegisterField("energy", &e_gf);
      pvdc.RegisterField("igr", &igr_gf);

      pvdc.SetCycle(cycle);
      pvdc.SetTime(time);
      pvdc.Save();

      if (pvdc.Error() != DataCollection::No_Error && Mpi::Root())
      {
         mfem::err << "Failed to write ParaView frame " << cycle << endl;
      }
   };

   if(paraview)
   {
      save_paraview_frame(para_vis_cycle, t, false);
      para_vis_cycle++;
      para_next_vis_time += para_vis_dt;
   }
   //   const double internal_energy = hydro.InternalEnergy(e_gf);
   //   const double kinetic_energy = hydro.KineticEnergy(v_gf);
   //   if (mpi.Root())
   //   {
   //      cout << std::fixed;
   //      cout << "step " << std::setw(5) << 0
   //            << ",\tt = " << std::setw(5) << std::setprecision(4) << t
   //            << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
   //            << ",\t|IE| = " << std::setprecision(10) << std::scientific
   //            << internal_energy
   //            << ",\t|KE| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy
   //            << ",\t|E| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy+internal_energy;
   //      cout << std::fixed;
   //      if (mem_usage)
   //      {
   //         cout << ", mem: " << mmax << "/" << msum << " MB";
   //      }
   //      cout << endl;
   //   }
   //

#ifdef LAGHOS_USE_CALIPER
   CALI_CXX_MARK_LOOP_BEGIN(mainloop_annotation, "timestep loop");
#endif
   int ti = 1;
   for (; !last_step; ti++)
   {
#ifdef LAGHOS_USE_CALIPER
      CALI_CXX_MARK_LOOP_ITERATION(mainloop_annotation, static_cast<int>(ti));
#endif
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      hydro.ResetTimeStepEstimate();

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(S, t, dt);
      steps++;

      // Adaptive time step control.
      const double dt_est = hydro.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         if (dt < std::numeric_limits<double>::epsilon())
         { MFEM_ABORT("The time step crashed!"); }
         t = t_old;
         S = S_old;
         hydro.ResetQuadratureData();
         if (Mpi::Root()) { cout << "Repeating step " << ti << endl; }
         if (steps < max_tsteps) { last_step = false; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }
      if(dtmax > 0 && dt >dtmax){ dt = dtmax;}
      if (dt_est < 1e-7) {
         if (Mpi::Root()) {
            cout << "WARNING: dt_est = " << dt_est
                 << " fell below 1e-7 at t = " << t
                 << ". Stopping early." << endl;
         }
         t_final = t;
         last_step = true;
      }
      if (requested_parabolic && !parabolic)
      {
         hydro.UpdateParabolic(true);
         parabolic = true;
         hydro.ResetQuadratureData();
      }

      // Ensure the sub-vectors x_gf, v_gf, and e_gf know the location of the
      // data in S. This operation simply updates the Memory validity flags of
      // the sub-vectors to match those of S.
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);
      igr_gf.SyncAliasMemory(S);

      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      pmesh.NewNodes(x_gf, false);

      if (last_step || (ti % vis_steps) == 0)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
         if (mem_usage)
         {
            mem = GetMaxRssMB();
            size_t mfree, mtot;
            if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
            {
               Device::DeviceMem(&mfree, &mtot);
               dmem = mtot - mfree;
               MPI_Reduce(&dmem, &dmmax, 1, MPI_LONG, MPI_MAX, 0,
                          pmesh.GetComm());
               MPI_Reduce(&dmem, &dmsum, 1, MPI_LONG, MPI_SUM, 0,
                          pmesh.GetComm());
               dmmax /= 1024*1024;
               dmsum /= 1024*1024;
            }
            MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh.GetComm());
            MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh.GetComm());
         }
         // const double internal_energy = hydro.InternalEnergy(e_gf);
         // const double kinetic_energy = hydro.KineticEnergy(v_gf);
         if (Mpi::Root())
         {
            const double sqrt_norm = sqrt(norm);

            cout << std::fixed;
            cout << "step " << std::setw(5) << ti
                 << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                 << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
                 << ",\t|e| = " << std::setprecision(10) << std::scientific
                 << sqrt_norm;
            //  << ",\t|IE| = " << std::setprecision(10) << std::scientific
            //  << internal_energy
            //   << ",\t|KE| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy
            //   << ",\t|E| = " << std::setprecision(10) << std::scientific
            //  << kinetic_energy+internal_energy;
            cout << std::fixed;
            if (mem_usage)
            {
               cout << ", mem: " << mmax << "/" << msum << " MB, "
                    << dmmax << "/" << dmsum << " MB";
            }
            cout << endl;
         }

         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh.GetComm());

         if (visualization || visit || gfprint) { hydro.ComputeDensity(rho_gf); }
         if (visualization && plot_pressure)
         {
            p_gf.ProjectCoefficient(pressure_coeff);
         }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets
            if (problem != 0 && problem != 4)
            {
               hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                             "Density", Wx, Wy, Ww, Wh);
            }
            Wx += offx;
            hydrodynamics::VisualizeField(vis_v, vishost, visport,
                                          v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                          "Specific Internal Energy",
                                          Wx, Wy, Ww,Wh);
            Wx += offx;
            if(useIGR){
               hydrodynamics::VisualizeField(vis_igr, vishost, visport, igr_gf,
                                       "IGR", Wx, Wy, Ww, Wh);
	            Wx += offx;
	         }
            if (plot_pressure)
            {
               hydrodynamics::VisualizeField(vis_p, vishost, visport, p_gf,
                                             "Pressure", Wx, Wy, Ww, Wh);
            }
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }

         if (gfprint && false)
         {
            std::ostringstream mesh_name, rho_name, v_name, e_name;
            mesh_name << basename << "_" << ti << "_mesh";
            rho_name  << basename << "_" << ti << "_rho";
            v_name << basename << "_" << ti << "_v";
            e_name << basename << "_" << ti << "_e";

            std::ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            pmesh.PrintAsOne(mesh_ofs);
            mesh_ofs.close();

            std::ofstream rho_ofs(rho_name.str().c_str());
            rho_ofs.precision(8);
            rho_gf.SaveAsOne(rho_ofs);
            rho_ofs.close();

            std::ofstream v_ofs(v_name.str().c_str());
            v_ofs.precision(8);
            v_gf.SaveAsOne(v_ofs);
            v_ofs.close();

            std::ofstream e_ofs(e_name.str().c_str());
            e_ofs.precision(8);
            e_gf.SaveAsOne(e_ofs);
            e_ofs.close();
         }
      }

      //ParaView Print
      if(paraview){
         if (t >= para_next_vis_time || ti == 0)
         {
            save_paraview_frame(para_vis_cycle, t, true);

            para_next_vis_time += para_vis_dt;
            para_vis_cycle++;
         }
      }

      // Problems checks
      if (check)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
         const double e_norm = sqrt(norm);
         MFEM_VERIFY(rs_levels == 0 && rp_levels == 0, "check: rs, rp");
         MFEM_VERIFY(order_v == 2, "check: order_v");
         MFEM_VERIFY(order_e == 1, "check: order_e");
         MFEM_VERIFY(ode_solver_type == 4, "check: ode_solver_type");
         MFEM_VERIFY(t_final == 0.6, "check: t_final");
         MFEM_VERIFY(cfl == 0.5, "check: cfl");
         MFEM_VERIFY(dim == 2 || dim == 3, "check: dimension");
         MFEM_VERIFY(std::string(mesh_file) == "data/square01_quad.mesh" ||
                     std::string(mesh_file) == "data/cube01_hex.mesh", "check: mesh_file");
         Checks(ti, e_norm, checks);
      }
   }

   if (gfprint)
   {
      std::ostringstream mesh_name, rho_name, v_name, e_name, igr_name;
      
      if(Mpi::Root()){std::cout << basename << igr_folder << alpha_folder << problem_folder << run_name << "\n";}
      mesh_name << basename << igr_folder << alpha_folder << problem_folder << run_name << "_mesh";
      rho_name  << basename << igr_folder << alpha_folder << problem_folder << run_name << "_rho";
      v_name << basename << igr_folder << alpha_folder << problem_folder << run_name << "_v";
      e_name << basename << igr_folder << alpha_folder << problem_folder << run_name << "_e";
      igr_name  << basename << igr_folder << alpha_folder << problem_folder << run_name << "_igr";
      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.PrintAsOne(mesh_ofs);
      mesh_ofs.close();
         

      ParGridFunction rho_h1(&H1FEScalarSpace), e_h1(&H1FEScalarSpace);
      rho_h1.ProjectGridFunction(rho_gf);
      e_h1.ProjectGridFunction(e_gf);

 
      std::ofstream rho_ofs(rho_name.str().c_str());
      rho_ofs.precision(8);
      rho_h1.SaveAsOne(rho_ofs);
      rho_ofs.close();

      std::ofstream v_ofs(v_name.str().c_str());
      v_ofs.precision(8);
      v_gf.SaveAsOne(v_ofs);
      v_ofs.close();

      std::ofstream e_ofs(e_name.str().c_str());
      e_ofs.precision(8);
      e_h1.SaveAsOne(e_ofs);
      e_ofs.close();

      std::ofstream igr_ofs(igr_name.str().c_str());
      igr_ofs.precision(8);
      igr_gf.SaveAsOne(igr_ofs);
      igr_ofs.close();
   }



#ifdef LAGHOS_USE_CALIPER
   CALI_CXX_MARK_LOOP_END(mainloop_annotation);
   adiak::value("steps", ti);
#endif

   //End timer
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;
   double elapsed_seconds = duration.count();
   if(Mpi::Root()){cout << "Execution time: " << elapsed_seconds << " seconds." << endl; }

   MFEM_VERIFY(!check || checks == 2, "Check error!");

   switch (ode_solver_type)
   {
      case 2: steps *= 2; break;
      case 3: steps *= 3; break;
      case 4: steps *= 4; break;
      case 6: steps *= 6; break;
      case 7: steps *= 2;
   }

   hydro.PrintTimingData(Mpi::Root(), steps, fom);

   if (mem_usage)
   {
      if (Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK))
      {
         size_t mfree, mtot;
         Device::DeviceMem(&mfree, &mtot);
         dmem = mtot - mfree;
         MPI_Reduce(&dmem, &dmmax, 1, MPI_LONG, MPI_MAX, 0, pmesh.GetComm());
         MPI_Reduce(&dmem, &dmsum, 1, MPI_LONG, MPI_SUM, 0, pmesh.GetComm());
         dmmax /= 1024*1024;
         dmsum /= 1024*1024;
      }
      mem = GetMaxRssMB();
      MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh.GetComm());
      MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh.GetComm());
   }

   const double energy_final = hydro.InternalEnergy(e_gf) +
                               hydro.KineticEnergy(v_gf);
   if (Mpi::Root())
   {
      cout << endl;
      cout << "Energy  diff: " << std::scientific << std::setprecision(2)
           << fabs(energy_init - energy_final) << endl;
      if (mem_usage)
      {
         cout << "Maximum memory resident set size: " << mmax << "/" << msum
              << " MB, " << dmmax << "/" << dmsum << " MB" << endl;
      }
   }

   // Print the error.
   // For problems 0 and 4 the exact velocity is constant in time.
   if (problem == 0 || problem == 4)
   {
      const double error_max = v_gf.ComputeMaxError(v_coeff),
                   error_l1  = v_gf.ComputeL1Error(v_coeff),
                   error_l2  = v_gf.ComputeL2Error(v_coeff);
      if (Mpi::Root())
      {
         cout << "L_inf  error: " << error_max << endl
              << "L_1    error: " << error_l1 << endl
              << "L_2    error: " << error_l2 << endl;
      }
   }

   if (visualization)
   {
      vis_v.close();
      vis_e.close();
      vis_p.close();
   }

#ifdef LAGHOS_USE_CALIPER
   adiak::fini();
#endif

#ifdef LAGHOS_USE_DEVICE_UMPIRE
   if (Mpi::Root())
   {
      cout << "Umpire device memory pool size: " << dev_pool_size << " GB" << endl;
      cout << "Umpire device memory high water mark: " << getDeviceMemoryHighWatermark() << " GB" << endl;
#ifdef LAGHOS_USE_CALIPER
      adiak::value("umpire_device_pool_size", dev_pool_size);
      adiak::value("umpire_device_high_water_mark", getDeviceMemoryHighWatermark());
#endif
   }
#endif

   if (check_exact_sedov)
   {
      // compare against the exact Sedov solution
      double gamma = 1.4;
      double rho0 = 1;
      double omega = 0;

      SedovSol asol(dim, gamma, rho0, blast_energy, omega);

      asol.SetTime(t_final);

      if (strncmp(mesh_file, "default", 7) == 0)
      {
         real_t min_r = std::min(std::min(Sx, Sy), Sz);
         MFEM_VERIFY(
            asol.r2 <= min_r,
            "Solution reflections off boundaries detected, cannot compare "
            "against exact solution.");
      }

      int err_order = std::max((std::max(order_v, order_e) + 1) * 2, order_q) * 2;
      const IntegrationRule &irule =
         IntRules.Get(pmesh.GetTypicalElementGeometry(), err_order);

      QuadratureSpace qspace(pmesh, irule);
      // only compare density
      QuadratureFunction sim_qfunc(qspace, 1);
      QuadratureFunction err_qfunc(qspace, 1);

      hydro.ComputeDensity(rho_gf);

      rho_gf.HostReadWrite();

      {
         GridFunctionCoefficient ctmp(&rho_gf);
         ctmp.Coefficient::Project(sim_qfunc);
      }

      auto slambda = [&](const Vector &x, Vector &res)
      {
         real_t tmp[3];
         Vector dr(tmp, dim);
         double r = 0;

         for (int i = 0; i < dim; ++i)
         {
            dr[i] = x[i] - blast_position[i];
            r += dr[i] * dr[i];
         }
         r = sqrt(r);
         if (r > 0)
         {
            for (int i = 0; i < dim; ++i)
            {
               dr[i] /= r;
            }
         }
         else
         {
            dr = 0_r;
         }
         double rho, v, P;
         asol.EvalSol(r, rho, v, P);
         res[0] = rho;
      };
      VectorFunctionCoefficient asol_coeff(1, slambda);
      asol_coeff.Project(err_qfunc);

      sim_qfunc.HostRead();
      err_qfunc.HostReadWrite();
      for (int i = 0; i < err_qfunc.Size(); ++i)
      {
         err_qfunc[i] = pow(err_qfunc[i] - AsConst(sim_qfunc)[i], 2);
      }
      real_t lrho_err = err_qfunc.Integrate();
      if (Mpi::Root())
      {
         cout << "Density L2 error: " << sqrt(lrho_err) << endl;
      }
   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}

double rho0(const Vector &x)
{
   switch (problem)
   {
      case 0: return 1.0;
      case 1: return 1.0;
      case 2: return (x(0) < 0.5) ? 1.0 : 0.1;
      case 3: return (dim == 2) ? (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0
                        : x(0) > 1.0 && ((x(1) < 1.5 && x(2) < 1.5) ||
                                         (x(1) > 1.5 && x(2) > 1.5)) ? 0.125 : 1.0;
      case 4: return 1.0;
      case 5:
      {
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.5313; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 0.8; }
         return 1.0;
      }
      case 6:
      {
         if (x(0) <  0.5 && x(1) >= 0.5) { return 2.0; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 3.0; }
         return 1.0;
      }
      case 7: return x(1) >= 0.0 ? 2.0 : 1.0;
      case 8: return 1.0;
	   case 9: return 1.0;
	   case 10: return 0.5*tanh(100*(0.5-x(0)))+.6;
      case 11: return 1.0;
      case 12: return (x(0) < 0.4) ? 1.0 : ((x(0) > 0.6) ? 0.1 : 1.0 - 4.5*(x(0) - 0.4));
      case 13: return 0.5*tanh(100*(0.5-x(0)))+.6;
      case 14: return ((x(0) < 0.4) ? 1.6*tanh(40*(0.2-x(0))) + 2.4 : 1 - 0.2*cos(50*(x(0)-0.4)));
      case 15:
      {
         double lambda = 0.5*tanh(sharpness*(x(0)-1.0))+0.5; // Left/ right "percentage" for convex combination
         if (dim == 3)
         {
            // Smooth version of the 3D problem 3 density.  On the right side,
            // rho = 0.125 when y and z are on the same side of 1.5, and rho =
            // 1.0 otherwise.
            const double lambda_y =
               0.5*tanh(sharpness*(x(1)-1.5)) + 0.5;
            const double lambda_z =
               0.5*tanh(sharpness*(x(2)-1.5)) + 0.5;
            const double same_yz = lambda_y*lambda_z +
                                   (1.0-lambda_y)*(1.0-lambda_z);
            const double rho_right = 1.0 - 0.875*same_yz;
            return lambda*rho_right + (1.0-lambda)*1.0;
         }
         return lambda*(0.4375*tanh(sharpness*(1.5-x(1))) + 0.5625) + (1-lambda)*(1.0);
      } 
      case 16: return (dim == 2) ? (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0
                        : x(0) > 1.0 && ((x(1) < 1.5 && x(2) < 1.5) ||
                                         (x(1) > 1.5 && x(2) > 1.5)) ? 0.125 : 1.0;
      case 17: return (dim == 2) ? (0.5 + 0.5*tanh(-200*sharpness*(x(0)-0.59375)))
                                 + (0.5 + 0.5*tanh(200*sharpness*(x(0)-0.59375)))*(0.4 - 0.2*tanh(200*sharpness*(x(1)-0.5)))
                        : 0.6 + 0.4*tanh(200*sharpness*(0.59375-x(0)));
      case 18:
      {
         double rho1 = 0.5;
         double rho2 = 0.2;
         double rho3 = 1.5;
         double lambda = 0.5 + 0.5*tanh(sharpness*(x(0)-2.5));
         return (1-lambda)*((rho1-rho2)/2.0*(1.0-tanh(sharpness*(x(0)-Sx/4))) + rho2) 
         + lambda*((rho2-rho3)/2.0*(1.0-tanh(sharpness*(x(0)-Sx*7/10+Sx/30*cos(2*M_PI*x(1)/Sy)))) + rho3);
      }
      case 19: return Case19Density(x);
      case 20: return 1.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

double gamma_func(const Vector &x)
{
   switch (problem)
   {
      case 0: return 5.0 / 3.0;
      case 1: return 1.4;
      case 2: return 1.4;
      case 3:
         if (dim == 1) { return (x(0) > 0.5) ? 1.4 : 1.5; }
         else { return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5; }
      case 4: return 5.0 / 3.0;
      case 5: return 1.4;
      case 6: return 1.4;
      case 7: return 5.0 / 3.0;
      case 8: return 5.0 / 3.0;
	   case 9: return 5.0 / 3.0;
	   case 10: return 1.4;
      case 11: return 1.4;
      case 12: return 1.4;
      case 13: return 1.4;
      case 14: return 1.4;
      case 15:
      {
         double lambda = 0.5*tanh(sharpness*(x(0)-1.0))+0.5; // Left/ right "percentage" for convex combination
         if (dim == 3)
         {
            // Problem 3 uses gamma = 1.4 below y = 1.5 on the right and
            // gamma = 1.5 everywhere else; it is independent of z.
            const double lambda_y =
               0.5*tanh(sharpness*(x(1)-1.5)) + 0.5;
            const double gamma_right = 1.4 + 0.1*lambda_y;
            return lambda*gamma_right + (1.0-lambda)*1.5;
         }
         return lambda*(0.05*tanh(sharpness*(x(1)-1.5)) + 1.45) + (1-lambda)*(1.5);
      } 
      case 16:
         if (dim == 1) { return (x(0) > 0.5) ? 1.4 : 1.5; }
         else { return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5; }
      case 17: return (dim == 2) ? (0.65 + 0.65*tanh(-200*sharpness*(x(0)-0.59375)))
                                 + (0.5 + 0.5*tanh(200*sharpness*(x(0)-0.59375)))*(1.3 + 0.1*tanh(200*sharpness*(x(1)-0.5)))
                       : 1.4 + 0.1*tanh(200*sharpness*(x(0)-0.59375));
      case 18: return 1.4; 
      case 19: return Case19Gamma(x);
      case 20: return 1.4;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static double rad(double x, double y) { return sqrt(x*x + y*y); }

void v0(const Vector &x, Vector &v)
{
   const double atn = dim!=1 ? pow((x(0)*(1.0-x(0))*4*x(1)*(1.0-x(1))*4.0),
                                   0.4) : 0.0;
   switch (problem)
   {
      case 0:
         v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
         v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
         if (x.Size() == 3)
         {
            v(0) *= cos(M_PI*x(2));
            v(1) *= cos(M_PI*x(2));
            v(2) = 0.0;
         }
         break;
      case 1: v = 0.0; break;
      case 2: v = 0.0; break;
      case 3: v = 0.0; break;
      case 4:
      {
         v = 0.0;
         const double r = rad(x(0), x(1));
         if (r < 0.2)
         {
            v(0) =  5.0 * x(1);
            v(1) = -5.0 * x(0);
         }
         else if (r < 0.4)
         {
            v(0) =  2.0 * x(1) / r - 5.0 * x(1);
            v(1) = -2.0 * x(0) / r + 5.0 * x(0);
         }
         else { }
         break;
      }
      case 5:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=0.7276*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.7276*atn; return; }
         MFEM_ABORT("Error in problem 5!");
         return;
      }
      case 6:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=-0.5*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=-0.5*atn; return;}
         MFEM_ABORT("Error in problem 6!");
         return;
      }
      case 7:
      {
         v = 0.0;
         v(1) = 0.02 * exp(-2*M_PI*x(1)*x(1)) * cos(2*M_PI*x(0));
         break;
      }
      case 8:
      {  
         v = 0.0;   // Shock 2
         if(x(0) < 0.1){
            v(0) = 0.0;
         } else if(x(0) < 0.2){
            v(0) = 10*x(0) - 1.0;
         } else if(x(0) < 0.35){
		      v(0) = 1.0;
	      } else if(x(0) < 0.45){
		      v(0) = 4.5 - 10*x(0);
	      }
         break;
      }
	   case 9: v = 0.0; break;
	   case 10: v = 0.0; break;
      case 11: v = 0.0; v(0) = sin(2*M_PI*x(0)); break;
      case 12: v = 0.0; break;
      case 13: v = 0.0; break;
      case 14:
      {   v = 0.0;
         //cout << length << std::endl;
         if (x(0) < 0.2 + (0.5 - Sx / 2) + 0.1 * Sx ) { 
                  v(0) = tanh(100*(x(0) - (0.5 - Sx / 2) - 0.06*Sx) / Sx ) + 1.0; 
         } else { v(0) = tanh(40.0*(0.2-x(0))) + 1.0; }
         break;
      }
      case 15: v = 0.0; break;
      case 16: v = 0.0; break;
      case 17:
      {
         v = 0.0;
         v(0) = tanh(50*(x(0) - 0.3)) + tanh(50*(0.5 - x(0)));
         break;
      }
      case 18:
      {
         v = 0.0;
         double Us = 1.0;
         v(0) = Us/2.0*(tanh(sharpness*(x(0) - Sx/8)/4)
                        - tanh(sharpness*(x(0) - Sx*0.55)));
         break;
      }
      case 19:
      {
         v = 0.0;
         v(0) = tanh(50*(x(0) - 0.3)) + tanh(50*(0.5 - x(0)));
         break;
      }
      case 20: v = 0.0; break;
      default: MFEM_ABORT("Bad number given for problem id!");
   }
}

double e0(const Vector &x)
{
   switch (problem)
   {
      case 0:
      {
         const double denom = 2.0 / 3.0;  // (5/3 - 1) * density.
         double val;
         if (x.Size() == 2)
         {
            val = 1.0 + (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) / 4.0;
         }
         else
         {
            val = 100.0 + ((cos(2*M_PI*x(2)) + 2) *
                           (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) - 2) / 16.0;
         }
         return val/denom;
      }
      case 1: return 0.0; // This case in initialized in main().
      case 2: return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma_func(x) - 1.0)
                        : 0.1 / rho0(x) / (gamma_func(x) - 1.0);
      case 3: return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma_func(x) - 1.0)
                        : 1.0 / rho0(x) / (gamma_func(x) - 1.0);
      case 4:
      {
         const double r = rad(x(0), x(1)), rsq = x(0) * x(0) + x(1) * x(1);
         const double gamma = 5.0 / 3.0;
         if (r < 0.2)
         {
            return (5.0 + 25.0 / 2.0 * rsq) / (gamma - 1.0);
         }
         else if (r < 0.4)
         {
            const double t1 = 9.0 - 4.0 * log(0.2) + 25.0 / 2.0 * rsq;
            const double t2 = 20.0 * r - 4.0 * log(r);
            return (t1 - t2) / (gamma - 1.0);
         }
         else { return (3.0 + 4.0 * log(2.0)) / (gamma - 1.0); }
      }
      case 5:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.4 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 5!");
         return 0.0;
      }
      case 6:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 6!");
         return 0.0;
      }
      case 7:
      {
         const double rho = rho0(x), gamma = gamma_func(x);
         return (6.0 - rho * x(1)) / (gamma - 1.0) / rho;
      }
      case 8: return 0.0;
	   case 9: return 0.0; // This case in initialized in main().
	   case 10: return 0.25;
      case 11: return 0.25;
      case 12: return 0.25*pow(( (x(0) < 0.4) ? 1.0 : ((x(0) > 0.6) ? 0.1 : 1.0 - 4.5*(x(0) - 0.4))),2.0);
      case 13: return 0.25*pow(0.5*tanh(100*(0.5-x(0)))+.6,2.0);
      case 14: return 2.5*( 4.5*tanh(40*(0.2-x(0))) + 5.5 )/ rho0(x);
      case 15:
      {
         double lambda = 0.5*tanh(sharpness*(x(0)-1.0))+0.5; // Left/ right "percentage" for convex combination
         if (dim == 3)
         {
            // Smoothly blend the four right-side values of problem 3:
            //
            //                z < 1.5   z > 1.5
            //   y < 1.5         2.0       0.25
            //   y > 1.5         0.2       1.6
            //
            // These are e = p/(rho*(gamma-1)) with p = 0.1.
            const double lambda_y =
               0.5*tanh(sharpness*(x(1)-1.5)) + 0.5;
            const double lambda_z =
               0.5*tanh(sharpness*(x(2)-1.5)) + 0.5;
            const double e_right =
               (1.0-lambda_y)*((1.0-lambda_z)*2.0 +
                               lambda_z*0.25) +
               lambda_y*((1.0-lambda_z)*0.2 + lambda_z*1.6);
            return lambda*e_right + (1.0-lambda)*2.0;
         }
         //return lambda*(0.1 / rho0(x) / (gamma_func(x) - 1.0)) + (1-lambda)*(2.0);
         return lambda*(0.675*tanh(sharpness*(x(1)-1.5)) + 0.925) + (1-lambda)*(2.0);
      } 
      case 16:
      { 
         

         double smooth_p = 0.55-0.45*tanh(sharpness*(x(0)-1.0));
         return smooth_p / (gamma_func(x) - 1.0) / rho0(x);
      }
      case 17:
      { 
         const double p0 = 1.0;
         return p0 / (gamma_func(x) - 1.0) / rho0(x);
      }
      case 18:
      {
         double phs = 4.5;
         double p0 = 1.0;
         double smooth_p = (phs-p0)/2.0*(1.0-tanh(sharpness*(x(0)-Sx/4))) + p0;
         return smooth_p / (gamma_func(x) - 1.0) / rho0(x);
      }
      case 19:
      {
         const double p0 = 1.0;
         return p0 / (gamma_func(x) - 1.0) / rho0(x);
      }
      case 20: return 0.0; // This case in initialized in main().
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static double Case19Transition(double x, double x0, double h)
{
   const double width = material_transition_points*h;
   if (width <= 0.0) { return (x < x0) ? 0.0 : 1.0; }

   const double t = (x - (x0 - 0.5*width)) / width;
   if (t <= 0.0) { return 0.0; }
   if (t >= 1.0) { return 1.0; }
   return t;
}

static double Case19Density(const Vector &x)
{
   const double sx = Case19Transition(x(0), 0.59375, material_mesh_h[0]);
   if (dim == 2)
   {
      const double sy = Case19Transition(x(1), 0.5, material_mesh_h[1]);
      const double right_density = (1.0 - sy)*0.6 + sy*0.2;
      return (1.0 - sx)*1.0 + sx*right_density;
   }
   return (1.0 - sx)*1.0 + sx*0.2;
}

static double Case19Gamma(const Vector &x)
{
   const double sx = Case19Transition(x(0), 0.59375, material_mesh_h[0]);
   if (dim == 2)
   {
      const double sy = Case19Transition(x(1), 0.5, material_mesh_h[1]);
      const double right_gamma = (1.0 - sy)*1.2 + sy*1.4;
      return (1.0 - sx)*1.3 + sx*right_gamma;
   }
   return (1.0 - sx)*1.3 + sx*1.5;
}

static void Case19EnergyBounds(double &emin, double &emax)
{
   emin = std::numeric_limits<double>::infinity();
   emax = 0.0;

   const double states_1d[][2] = {{1.0, 1.3}, {0.2, 1.5}};
   const double states_2d[][2] = {{1.0, 1.3}, {0.6, 1.2}, {0.2, 1.4}};
   const double (*states)[2] = (dim == 2) ? states_2d : states_1d;
   const int num_states = (dim == 2) ? 3 : 2;

   for (int i = 0; i < num_states; i++)
   {
      const double rho = states[i][0];
      const double gamma = states[i][1];
      const double e = 1.0 / ((gamma - 1.0)*rho);
      emin = std::min(emin, e);
      emax = std::max(emax, e);
   }
}

static void BoundGridFunction(ParGridFunction &gf, double min_val, double max_val)
{
   gf.HostReadWrite();
   for (int i = 0; i < gf.Size(); i++)
   {
      if (gf(i) < min_val) { gf(i) = min_val; }
      else if (gf(i) > max_val) { gf(i) = max_val; }
   }
}

static void SetMaterialMeshSpacing(const ParMesh &pmesh)
{
   const double inf = std::numeric_limits<double>::infinity();
   double local_min[3] = {inf, inf, inf};
   Array<int> vertices;

   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      pmesh.GetElementVertices(e, vertices);
      double min_coord[3] = {inf, inf, inf};
      double max_coord[3] = {-inf, -inf, -inf};

      for (int i = 0; i < vertices.Size(); i++)
      {
         const real_t *coord = pmesh.GetVertex(vertices[i]);
         for (int d = 0; d < pmesh.Dimension(); d++)
         {
            min_coord[d] = std::min(min_coord[d], coord[d]);
            max_coord[d] = std::max(max_coord[d], coord[d]);
         }
      }

      for (int d = 0; d < pmesh.Dimension(); d++)
      {
         const double extent = max_coord[d] - min_coord[d];
         if (extent > 0.0) { local_min[d] = std::min(local_min[d], extent); }
      }
   }

   double global_min[3];
   MPI_Allreduce(local_min, global_min, 3, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());

   for (int d = 0; d < pmesh.Dimension(); d++)
   {
      if (std::isfinite(global_min[d]) && global_min[d] > 0.0)
      {
         material_mesh_h[d] = global_min[d];
      }
   }
}

static void display_banner(std::ostream &os)
{
   os << endl
      << "       __                __                 " << endl
      << "      / /   ____  ____  / /_  ____  _____   " << endl
      << "     / /   / __ `/ __ `/ __ \\/ __ \\/ ___/ " << endl
      << "    / /___/ /_/ / /_/ / / / / /_/ (__  )    " << endl
      << "   /_____/\\__,_/\\__, /_/ /_/\\____/____/  " << endl
      << "               /____/                       " << endl << endl;
}

#ifdef LAGHOS_USE_CALIPER
static void RecordAdiakMetadata(int dim, const char *mesh_file, int elem_per_mpi,
                                int nx, int ny, int nz, double blast_energy,
                                double Sx, double Sy, double Sz,
                                int rs_levels, int rp_levels, int problem,
                                int order_v, int order_e, int order_q,
                                int ode_solver_type, double t_final, double cfl,
                                double cg_tol, double ftz_tol, double delta_tol,
                                int cg_max_iter, int max_tsteps,
                                bool p_assembly, bool impose_visc,
                                bool visualization, int vis_steps, bool visit,
                                bool gfprint, const char *basename,
                                const char *device, bool check,
                                bool check_exact_sedov, bool mem_usage,
                                bool fom, bool gpu_aware_mpi, int dev_pool_size,
                                bool enable_nc, int dev)
{
   adiak::value("dimension", dim);
   adiak::value("mesh", std::string(mesh_file));
   adiak::value("elem-per-mpi", elem_per_mpi);
   adiak::value("xelems", nx);
   adiak::value("yelems", ny);
   adiak::value("zelems", nz);
   adiak::value("blast-energy", blast_energy);
   adiak::value("xwidth", (double)Sx);
   adiak::value("ywidth", (double)Sy);
   adiak::value("zwidth", (double)Sz);
   adiak::value("refine-serial", rs_levels);
   adiak::value("refine-parallel", rp_levels);
   adiak::value("problem", problem);
   adiak::value("order-kinematic", order_v);
   adiak::value("order-thermo", order_e);
   adiak::value("order-intrule", order_q);
   adiak::value("ode-solver", ode_solver_type);
   adiak::value("t-final", t_final);
   adiak::value("cfl", cfl);
   adiak::value("cg-tol", cg_tol);
   adiak::value("ftz-tol", ftz_tol);
   adiak::value("delta-tol", delta_tol);
   adiak::value("cg-max-steps", cg_max_iter);
   adiak::value("max-steps", max_tsteps);
   adiak::value("partial-assembly", (int)p_assembly);
   adiak::value("impose-viscosity", (int)impose_visc);
   adiak::value("visualization", (int)visualization);
   adiak::value("visualization-steps", vis_steps);
   adiak::value("visit", (int)visit);
   adiak::value("print", (int)gfprint);
   adiak::value("outputfilename", std::string(basename));
   adiak::value("device", std::string(device));
   adiak::value("checks", (int)check);
   adiak::value("exact-error", (int)check_exact_sedov);
   adiak::value("mem", (int)mem_usage);
   adiak::value("fom", (int)fom);
   adiak::value("gpu-aware-mpi", (int)gpu_aware_mpi);
   adiak::value("dev-pool-size", dev_pool_size);
   adiak::value("conforming", (int)enable_nc);
   adiak::value("dev", dev);
}
#endif

static long GetMaxRssMB()
{
   struct rusage usage;
   if (getrusage(RUSAGE_SELF, &usage)) { return -1; }
#ifndef __APPLE__
   const long unit = 1024; // kilo
#else
   const long unit = 1024*1024; // mega
#endif
   return usage.ru_maxrss/unit; // mega bytes
}

static void Checks(const int ti, const double nrm, int &chk)
{
   const double eps = 1.e-13;

   auto check = [&](int p, int i, const double res)
   {
      auto rerr = [](const double a, const double v, const double eps)
      {
         MFEM_VERIFY(fabs(a) > eps && fabs(v) > eps, "One value is near zero!");
         const double err_a = fabs((a-v)/a);
         const double err_v = fabs((a-v)/v);
         return fmax(err_a, err_v) < eps;
      };
      if (problem == p && ti == i)
      {
         chk++;
         if (!rerr(nrm, res, eps))
         {
            printf("\033[33m%.15e\033[m\n",nrm);
         }
         MFEM_VERIFY(rerr(nrm, res, eps), "P"<<problem<<", #"<<i);
      }
   };

   const double it_norms[2][8][2][2] = // dim, problem, {it,norm}
   {
      {
         {{5, 6.546538624534384e+00}, { 27, 7.588576357792927e+00}},
         {{5, 3.508254945225794e+00}, { 15, 2.756444596823211e+00}},
         {{5, 1.020745795651244e+01}, { 59, 1.721590205901898e+01}},
         {{5, 8.000000000000000e+00}, { 16, 8.000000000000000e+00}},
         {{5, 3.446324942352448e+01}, { 18, 3.446844033767240e+01}},
         {{5, 1.030899557252528e+01}, { 36, 1.057362418574309e+01}},
         {{5, 8.039707010835693e+00}, { 36, 8.316970976817373e+00}},
         {{5, 1.514929259650760e+01}, { 25, 1.514931278155159e+01}},
      },
      {
         {{5, 1.198510951452527e+03}, {188, 1.199384410059154e+03}},
         {{5, 6.695818592962833e+00}, { 20, 4.267902387082487e+00}},
         {{5, 2.041491591302486e+01}, { 59, 3.443180411803796e+01}},
         {{5, 1.600000000000000e+01}, { 16, 1.600000000000000e+01}},
         {{5, 6.892649884704898e+01}, { 18, 6.893688067534482e+01}},
         {{5, 2.061984481890964e+01}, { 36, 2.114519664792607e+01}},
         {{5, 1.607988713996459e+01}, { 36, 1.662736010353023e+01}},
         {{5, 3.029858112572883e+01}, { 24, 3.029858832743707e+01}}
      }
   };

   for (int p=0; p<8; p++)
   {
      for (int i=0; i<2; i++)
      {
         const int it = static_cast<int>(it_norms[dim-2][p][i][0]);
         const double norm = it_norms[dim-2][p][i][1];
         check(p, it, norm);
      }
   }
}

static void AssignMeshBdrAttrs2D(Mesh& mesh, real_t xmin, real_t xmax)
{
   Vector pos(3);
   constexpr real_t tol = 1e-6;
   const int NBE = mesh.GetNBE();
   IntegrationPoint center;
   center.x = 0.5;
   center.y = 0.5;
   center.z = 0.5;
   for (int b = 0; b < NBE; b++)
   {
      Element *bel = mesh.GetBdrElement(b);
      auto eltrans = mesh.GetBdrElementTransformation(b);
      eltrans->Transform(center, pos);
      int attr = 2;
      if (pos[0] <= xmin + tol || pos[0] >= xmax - tol)
      {
         attr = 1;
      }
      bel->SetAttribute(attr);
   }
}

static void AssignMeshBdrAttrs3D(Mesh &mesh, real_t xmin, real_t xmax,
                                 real_t ymin, real_t ymax)
{
   Vector pos(3);
   constexpr real_t tol = 1e-6;
   const int NBE = mesh.GetNBE();
   IntegrationPoint center;
   center.x = 0.5;
   center.y = 0.5;
   center.z = 0.5;
   for (int b = 0; b < NBE; b++)
   {
      Element *bel = mesh.GetBdrElement(b);
      auto eltrans = mesh.GetBdrElementTransformation(b);
      eltrans->Transform(center, pos);
      int attr = 3;
      if (pos[0] <= xmin + tol || pos[0] >= xmax - tol)
      {
         attr = 1;
      }
      else if (pos[1] <= ymin + tol || pos[1] >= ymax - tol)
      {
         attr = 2;
      }
      bel->SetAttribute(attr);
   }
}
