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
//    p = 9  --> Gaussian Blast with partial refinement option
//    p = 10 --> Smooth (tanh) Sod shock tube
//    p = 11 --> Mach Number / sin init vel
//    p = 12 --> LeBlanc Shock Tube
//    p = 13 --> Smooth LeBlanc Tube
//    p = 14 --> Smooth Shu-Osher shock tube
//
// Sample runs: see README.md, section 'Verification of Results'.
//
// Combinations resulting in 3D uniform Cartesian MPI partitionings of the mesh:
// -m data/cube01_hex.mesh   -pt 211 for  2 / 16 / 128 / 1024 ... tasks.
// -m data/cube_922_hex.mesh -pt 921 for    / 18 / 144 / 1152 ... tasks.
// -m data/cube_522_hex.mesh -pt 522 for    / 20 / 160 / 1280 ... tasks.
// -m data/cube_12_hex.mesh  -pt 311 for  3 / 24 / 192 / 1536 ... tasks.
// -m data/cube01_hex.mesh   -pt 221 for  4 / 32 / 256 / 2048 ... tasks.
// -m data/cube_922_hex.mesh -pt 922 for    / 36 / 288 / 2304 ... tasks.
// -m data/cube_522_hex.mesh -pt 511 for  5 / 40 / 320 / 2560 ... tasks.
// -m data/cube_12_hex.mesh  -pt 321 for  6 / 48 / 384 / 3072 ... tasks.
// -m data/cube01_hex.mesh   -pt 111 for  8 / 64 / 512 / 4096 ... tasks.
// -m data/cube_922_hex.mesh -pt 911 for  9 / 72 / 576 / 4608 ... tasks.
// -m data/cube_522_hex.mesh -pt 521 for 10 / 80 / 640 / 5120 ... tasks.
// -m data/cube_12_hex.mesh  -pt 322 for 12 / 96 / 768 / 6144 ... tasks.

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>
#include <cmath>
#include <chrono>
#include "laghos_solver.hpp"
#include "laghos_IGR_solver.hpp"
#include "../mfem/fem/gslib.hpp"


using std::cout;
using std::endl;
using namespace std;
using namespace mfem;

// Choice for the problem setup.
static int problem, dim;
static double length;


// Forward declarations.
double e0(const Vector &);
double rho0(const Vector &);
double gamma_func(const Vector &);
void v0(const Vector &, Vector &);

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
   double var, h;

public:
   Gaussian(double var_, double h_)
      : var(var_), h(h_) {}

   virtual double Eval(mfem::ElementTransformation &T,
                       const mfem::IntegrationPoint &ip)
   {
      mfem::Vector x;
      T.Transform(ip, x);   // x = physical coordinates
	  int dim = T.GetSpaceDim();
	  double out = 1.0;
	  for(int i=0; i<dim; i++){
		  out *= h*Gauss((x[i] - 0.5) / sqrt(2*var)) / sqrt(2*3.141592*var);
	  }
      return out;
   }
};

int main(int argc, char *argv[])
{  
   //Start Timer
   auto start = chrono::high_resolution_clock::now();

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
   int rs_levels = 2;
   int rp_levels = 0;
   int rb_levels = 0;
   Array<int> cxyz;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   int ode_solver_type = 4;
   double t_final = 0.6;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   double ftz_tol = 0.0;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   bool p_assembly = false;
   bool impose_visc = false;
   bool visualization = false;
   bool write = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   bool gfread = false;
   const char *basename = "results/";
   int partition_type = 0;
   const char *device = "cpu";
   bool check = false;
   bool mem_usage = false;
   bool fom = false;
   bool gpu_aware_mpi = false;
   int dev = 0;
   double blast_energy = 0.25;
   length = 1.0;
   double blast_position[] = {0.5, 0.5, 0.5};
   double alpha = 0.001;
   double stallIGR = -0.3;
   double e_reg = 1.0;
   bool useIGR = true;
   bool corner = false;
   double variance = 0.1;
   double visc_const = 250;  // Viscosity constant i.e. A
   int visc_type = 3;  // 1 is Laghos Artificial Visc, 2 is const A, 3 is dx(A ||u|| + c)
   bool TestPrint = false;

   bool enable_nc = true;
   bool enable_rebalance = true;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&rb_levels, "-rb", "--refine-blast",
                  "Number of times to refine the mesh in serial around the blast.");
   args.AddOption(&cxyz, "-c", "--cartesian-partitioning",
                  "Use Cartesian partitioning.");
   args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_q, "-oq", "--order-intrule",
                  "Order  of the integration rule.");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Alpha as the level of IGR");
   args.AddOption(&blast_energy, "-be", "--blast-energy",
                  "Amplitude of shock/ sine");
   args.AddOption(&length, "-len", "--mesh-length",
                  "Mesh Length");
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
   args.AddOption(&visc_const, "-vc", "--visc-const",
                  "Sets the viscosity constant.");
   args.AddOption(&visc_type, "-vt", "--visc-type",
                  "Sets the way we add viscosity.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&write, "-w", "--write", "-no-w",
                  "--no-write",
                  "Enable or disable writing the grid function.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&gfread, "-read", "--read", "-no-read", "--no-read",
                  "Enable or disable reading the grid function.");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&partition_type, "-pt", "--partition",
                  "Customized x/y/z Cartesian MPI partitioning of the serial mesh.\n\t"
                  "Here x,y,z are relative task ratios in each direction.\n\t"
                  "Example: with 48 mpi tasks and -pt 321, one would get a Cartesian\n\t"
                  "partition of the serial mesh by (6,4,2) MPI tasks in (x,y,z).\n\t"
                  "NOTE: the serially refined mesh must have the appropriate number\n\t"
                  "of zones in each direction, e.g., the number of zones in direction x\n\t"
                  "must be divisible by the number of MPI tasks in direction x.\n\t"
                  "Available options: 11, 21, 111, 211, 221, 311, 321, 322, 432.");
   args.AddOption(&device, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&check, "-chk", "--checks", "-no-chk", "--no-checks",
                  "Enable 2D checks.");
   args.AddOption(&mem_usage, "-mb", "--mem", "-no-mem", "--no-mem",
                  "Enable memory usage.");
   args.AddOption(&fom, "-f", "--fom", "-no-fom", "--no-fom",
                  "Enable figure of merit output.");
   args.AddOption(&gpu_aware_mpi, "-gam", "--gpu-aware-mpi", "-no-gam",
                  "--no-gpu-aware-mpi", "Enable GPU aware MPI communications.");
   args.AddOption(&enable_nc, "-nc", "--nonconforming", "-no-nc",
                  "--conforming",
                  "Use non-conforming meshes. Requires a 2D or 3D mesh.");
   args.AddOption(&enable_rebalance, "-b", "--balance", "-no-b",
                  "--no-rebalance",
                  "Perform a rebalance after parallel refinement. Only enabled \n\t"
                  "for non-conforming meshes with Metis partitioning.");
   args.AddOption(&dev, "-dev", "--dev", "GPU device to use.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   // Configure the device from the command line options
   Device backend;
   backend.Configure(device, dev);
   if (Mpi::Root()) { backend.Print(); }
   backend.SetGPUAwareMPI(gpu_aware_mpi);

   //Read in grid functions
   GridFunction vBgf, eBgf, rhoBgf;
   H1_FECollection *H1FECser = nullptr;
   L2_FECollection *L2FECser = nullptr;
   FiniteElementSpace *L2FESpaceSer = nullptr;
   FiniteElementSpace *H1FESpaceSer = nullptr;

   // On all processors, use the default builtin 1D/2D/3D mesh or read the
   // serial one given on the command line.
   Mesh *mesh;
   if (strncmp(mesh_file, "default", 7) != 0)
   {
      mesh = new Mesh(mesh_file, true, true);
   }
   else
   {
      if (dim == 1)
      {
         if(gfread){
            std::ifstream file("data.csv");
            if (!file.is_open()) {
               MFEM_ABORT("Could not open file!");
            }

            std::string line;

            // --- Read and ignore header ---
            std::getline(file, line);

            std::vector<double> xB, rhoB, muB, EB, vB, pB, SigmaB, eB;

            // --- Read data rows ---
            while (std::getline(file, line)) {
               std::stringstream ss(line);
               std::string field;

               std::getline(ss, field, ',');
               xB.push_back(std::stod(field));

               std::getline(ss, field, ',');
               rhoB.push_back(std::stod(field));

               std::getline(ss, field, ',');
               muB.push_back(std::stod(field));

               std::getline(ss, field, ',');
               EB.push_back(std::stod(field));

               std::getline(ss, field, ',');
               vB.push_back(std::stod(field));

               std::getline(ss, field, ',');
               pB.push_back(std::stod(field));

               std::getline(ss, field, ',');
               SigmaB.push_back(std::stod(field));

               std::getline(ss, field, ',');
               eB.push_back(std::stod(field));

               std::getline(ss, field, ',');
               SigmaB.push_back(std::stod(field));
            }

            std::cout << "Read " << xB.size() << " rows\n";

            int scale = pow(2, rs_levels);
            int UnscaledSize = xB.size();

            mesh = new Mesh(Mesh::MakeCartesian1D(scale*(xB.size()-1)));
            mesh->GetBdrElement(0)->SetAttribute(1);
            mesh->GetBdrElement(1)->SetAttribute(1);

            //MFEM_VERIFY(xB.size() == mesh->GetNV(), "xB size must match number of mesh vertices");

            

            for (int i = 0; i < UnscaledSize-1; i++)
            { 
               for(int j = 0; j < scale; j++){
                  mesh->GetVertex(scale*i + j)[0] = (scale - j)*xB[i]/scale + j*xB[i+1]/scale;
               }  
            }
            mesh->GetVertex(mesh->GetNV()-1)[0] = xB[xB.size()-1]; //Set right boundary values

            H1_FECollection fec_p1(1, mesh->Dimension());
            FiniteElementSpace fes_p1(mesh, &fec_p1);
            MFEM_VERIFY(fes_p1.GetNDofs() == mesh->GetNV(), "H1 P1 DOFs should equal number of vertices");

            GridFunction v_p1(&fes_p1), e_p1(&fes_p1), rho_p1(&fes_p1);
            MFEM_VERIFY(scale*rhoB.size()-(scale-1) == rho_p1.Size(), "rhoB size must match H1 P1 DOFs");

            //Set grid function values
            for (int i = 0; i < UnscaledSize-1; i++)
            {
               for(int j = 0; j < scale; j++){
                  v_p1(scale*i + j) = (scale - j)*vB[i]/scale + j*vB[i+1]/scale;
                  e_p1(scale*i + j) = (scale - j)*eB[i]/scale + j*eB[i+1]/scale;
                  rho_p1(scale*i + j) = (scale - j)*rhoB[i]/scale + j*rhoB[i+1]/scale;
               }
            }
            v_p1(rho_p1.Size()-1) = vB[UnscaledSize-1]; //Set right boundary values
            e_p1(rho_p1.Size()-1) = eB[UnscaledSize-1]; 
            rho_p1(rho_p1.Size()-1) = rhoB[UnscaledSize-1];

            //Serial Projection of read values
            L2FECser = new L2_FECollection(order_e, dim, BasisType::Positive); //Exactly the same as below
            H1FECser = new H1_FECollection(order_v, dim);
            L2FESpaceSer = new FiniteElementSpace(mesh, L2FECser);
            H1FESpaceSer = new FiniteElementSpace(mesh, H1FECser, mesh->Dimension());

            vBgf.SetSpace(H1FESpaceSer);
            eBgf.SetSpace(L2FESpaceSer);
            rhoBgf.SetSpace(L2FESpaceSer);
            
            rhoBgf.ProjectGridFunction(rho_p1);
            vBgf.ProjectGridFunction(v_p1);
            eBgf.ProjectGridFunction(e_p1);

         } else {
            mesh = new Mesh(Mesh::MakeCartesian1D(2, length));
            mesh->GetBdrElement(0)->SetAttribute(1);
            mesh->GetBdrElement(1)->SetAttribute(1);
            if(length != 1.0){
               for (int i = 0; i < mesh->GetNV(); i++)
                  { mesh->GetVertex(i)[0] -= (length - 1.0) / 2.0; }
            }
         }
      }
      if (dim == 2)
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/2) ? 2 : 1;
            bel->SetAttribute(attr);
         }
      }
      if (dim == 3)
      {
         mesh = new Mesh(Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/3) ? 3 : (b < 2*NBE/3) ? 1 : 2;
            bel->SetAttribute(attr);
         }
      }
   }
   dim = mesh->Dimension();

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
      mesh->EnsureNCMesh();
   }

   // Refine the mesh in serial to increase the resolution.
   if(!gfread || dim > 1){for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }}
   if(dim == 1 && myid == 0){cout << "Serial dx = " << (length / (mesh->GetNV()-1)) << std::endl;}
   if(dim == 1 && myid == 0){cout << "Serial dx^2 = " << (length / (mesh->GetNV()-1))*(length / (mesh->GetNV()-1)) << std::endl;}
   if(dim == 2 && myid == 0){cout << "Serial dx^2 is about " << length*length / mesh->GetNV() << std::endl;}
   const int mesh_NE = mesh->GetNE();
   if (Mpi::Root())
   {
      cout << "Number of zones in the serial mesh: " << mesh_NE << endl;
   }

   




   
   if(problem == 9){
	   
      for (int level = 0; level < rb_levels; level++)
      {
         Array<int> el_to_refine;
         el_to_refine.Reserve(mesh->GetNE());

         for (int i = 0; i < mesh->GetNE(); i++)
         {
           ElementTransformation *T = mesh->GetElementTransformation(i);

            IntegrationPoint ip; ip.Set2(1.0/2.0, 1.0/2.0);
            Vector x(dim);
            T->Transform(ip, x);

            double xx = x(0), yy = x(1);

            bool near_blast = false;
            {
               double pad = 1.5*(3-level)*sqrt(variance);
               near_blast = (abs(xx - 0.5) <= pad && abs(yy - 0.5) <=  pad);
            }

            if (near_blast) { el_to_refine.Append(i); }
         }

         if (el_to_refine.Size() > 0)
         {
            mesh->GeneralRefinement(el_to_refine); // triangles: conforming refinement
         }
      } 
      mesh->EnsureNodes();
	   
   }
   
   

   // Parallel partitioning of the mesh.
   ParMesh *pmesh = nullptr;
   const int num_tasks = Mpi::WorldSize(); int unit = 1;
   int *nxyz = new int[dim];
   switch (partition_type)
   {
      case 0:
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 11:
      case 111:
         unit = static_cast<int>(floor(pow(num_tasks, 1.0 / dim) + 1e-2));
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 21: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 2) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit;
         break;
      case 31: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit;
         break;
      case 32: // 2D
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit;
         break;
      case 49: // 2D
         unit = static_cast<int>(floor(pow(9 * num_tasks / 4, 1.0 / 2) + 1e-2));
         nxyz[0] = 4 * unit / 9; nxyz[1] = unit;
         break;
      case 51: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 2) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit;
         break;
      case 211: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 221: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 4, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 311: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 321: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 6, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 322: // 3D.
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 432: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 3 * unit / 2; nxyz[2] = unit;
         break;
      case 511: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 521: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 10, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 522: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 20, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
         break;
      case 911: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 9, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 921: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 18, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 922: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 36, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
         break;
      default:
         if (myid == 0)
         {
            cout << "Unknown partition type: " << partition_type << '\n';
         }
         delete mesh;
         MPI_Finalize();
         return 3;
   }
   int product = 1;
   for (int d = 0; d < dim; d++) { product *= nxyz[d]; }
   const bool cartesian_partitioning = (cxyz.Size()>0)?true:false;
   if (product == num_tasks || cartesian_partitioning)
   {
      if (cartesian_partitioning)
      {
         int cproduct = 1;
         for (int d = 0; d < dim; d++) { cproduct *= cxyz[d]; }
         MFEM_VERIFY(!cartesian_partitioning || cxyz.Size() == dim,
                     "Expected " << mesh->SpaceDimension() << " integers with the "
                     "option --cartesian-partitioning.");
         MFEM_VERIFY(!cartesian_partitioning || num_tasks == cproduct,
                     "Expected cartesian partitioning product to match number of ranks.");
      }
      int *partitioning = cartesian_partitioning ?
                          mesh->CartesianPartitioning(cxyz):
                          mesh->CartesianPartitioning(nxyz);
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
      delete [] partitioning;
   }
   else
   {
      if (myid == 0)
      {
         cout << "Non-Cartesian partitioning through METIS will be used.\n";
#ifndef MFEM_USE_METIS
         cout << "MFEM was built without METIS. "
              << "Adjust the number of tasks to use a Cartesian split." << endl;
#endif
      }
#ifndef MFEM_USE_METIS
      return 1;
#endif
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   }
   delete [] nxyz;
   delete mesh;

   // Refine the mesh further in parallel to increase the resolution.
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   if (!cartesian_partitioning && enable_nc && dim > 1)
   {
      if (myid == 0) { cout << "Rebalancing mesh" << endl; }
      pmesh->Rebalance();
   }

   int NE = pmesh->GetNE(), ne_min, ne_max;
   MPI_Reduce(&NE, &ne_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&NE, &ne_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   { cout << "Zones min/max: " << ne_min << " " << ne_max << endl; }

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());
   ParFiniteElementSpace H1FEScalarSpace(pmesh, &H1FEC, 1);

   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs, ess_vdofs;
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), dofs_marker, dofs_list;
      for (int d = 0; d < pmesh->Dimension(); d++)
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
   ODESolver *ode_solver = NULL;
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
         delete pmesh;
         MPI_Finalize();
         return 3;
   }

   const HYPRE_BigInt glob_size_l2 = L2FESpace.GlobalTrueVSize();
   const HYPRE_BigInt glob_size_h1 = H1FESpace.GlobalTrueVSize();
   const HYPRE_BigInt glob_size_h1_scal = H1FEScalarSpace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of kinematic (position, velocity) dofs: "
           << glob_size_h1 << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_l2 << "\n\n" << endl;
   }

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy
   const int Vsize_l2 = L2FESpace.GetVSize();
   const int Vsize_h1 = H1FESpace.GetVSize();
   const int Vsize_h12 = H1FESpace.GetVSize();
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
   pmesh->SetNodalGridFunction(&x_gf);
   // Sync the data location of x_gf with its base, S
   x_gf.SyncAliasMemory(S);

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   real_t Mach = (problem == 11) ? blast_energy : 1.0;
   ScalarVectorProductCoefficient v_coeff_scaled(Mach, v_coeff);

   v_gf.ProjectCoefficient(v_coeff_scaled);
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
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes), l2_one(&l2_fes);
   l2_rho0_gf.ProjectCoefficient(rho0_coeff);
   rho0_gf.ProjectGridFunction(l2_rho0_gf);
   
   
   if(corner){for(int i=0; i<3; i++){blast_position[i] = 0.0;}} //Else in the center of the inline quad
   if (problem == 1)
   {  
      ConstantCoefficient reg(e_reg);
	  l2_one.ProjectCoefficient(reg);
      DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                               blast_position[2], blast_energy);
      l2_e.ProjectCoefficient(e_coeff);
      l2_e += l2_one;
   }
   else if(problem == 9){
	  // For the Sedov test, we use a delta function at the origin.
	  ConstantCoefficient reg(e_reg);
	  l2_one.ProjectCoefficient(reg);
	  
	  Gaussian smoothblast(variance, blast_energy);
      l2_e.ProjectCoefficient(smoothblast);
	  l2_e += l2_one;
   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);
   // Sync the data location of e_gf with its base, S
   e_gf.SyncAliasMemory(S);

   igr_gf = 0.0;
   igr_gf.SyncAliasMemory(S);

   //cout << "Everything ok so far 1" << endl;

   //Set initial conditions from read in values
   if(gfread && dim == 1){
      //GridFunctionCoefficient vBcoeff(&vBgf), eBcoeff(&eBgf), rhoBcoeff(&rhoBgf);
      rho0_gf = rhoBgf;
      e_gf = eBgf; e_gf.SyncAliasMemory(S);
      v_gf = vBgf; v_gf.SyncAliasMemory(S);
   }

   //cout << "Everything ok so far 2" << endl;

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on function that's constant on the moving mesh.
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma_func);
   mat_gf.ProjectCoefficient(mat_coeff);

   // Additional details, depending on the problem.
   int source = 0; bool visc = true, vorticity = false;
   switch (problem)
   {
      case 0: if (pmesh->Dimension() == 2) { source = 1; } visc = false; break;
      case 1: visc = true; break;
      case 2: visc = true; break;
      case 3: visc = true; S.HostRead(); break;
      case 4: visc = false; break;
      case 5: visc = true; break;
      case 6: visc = true; break;
      case 7: source = 2; visc = true; vorticity = true;  break;
	  case 8: visc = true; break;
	  case 9: visc = false; break;
	  case 10: visc = false; break;
     case 11: visc = false; break;
     case 12: visc = false; break;
     case 13: visc = false; break;
     case 14: visc = false; break;
      default: MFEM_ABORT("Wrong problem specification!");
   }
   if (impose_visc || visc_const > 0) { visc = true; }
   bool visc_igr = impose_visc;

   hydrodynamics::LagrangianIGRHydroOperator hydro(S.Size(),
                                                H1FESpace, H1FEScalarSpace, L2FESpace, ess_tdofs,
                                                rho0_coeff, rho0_gf,
                                                mat_gf, source, cfl,
                                                visc, vorticity, p_assembly,
                                                cg_tol, cg_max_iter, ftz_tol,
                                                order_q, useIGR);
   hydro.SetAlpha(alpha);
   if(visc_const > 0){ hydro.SetViscConst(visc_const); }
   hydro.SetViscType(visc_type);

   socketstream vis_rho, vis_v, vis_e, vis_igr;
   char vishost[] = "localhost";
   int  visport   = 19916;

   ParGridFunction rho_gf;
   if (visualization || visit) { hydro.ComputeDensity(rho_gf); }
   const double energy_init = hydro.InternalEnergy(e_gf) +
                              hydro.KineticEnergy(v_gf);

   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      vis_igr.precision(8);
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
   }

   // Save data for VisIt visualization.
   VisItDataCollection visit_dc(basename, pmesh);
   if (visit)
   {
      visit_dc.RegisterField("Density",  &rho_gf);
      visit_dc.RegisterField("Velocity", &v_gf);
      visit_dc.RegisterField("Specific Internal Energy", &e_gf);
      visit_dc.RegisterField("IGR", &igr_gf);
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
   int checks = 0;
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
   for (int ti = 1; !last_step; ti++)
   {  
      if(t < stallIGR){
		  hydro.UpdateUseVisc(true);
		  hydro.UpdateUseIGR(false);
	  } else if(t < stallIGR + 0.2){
		  hydro.UpdateUseVisc(true);
		  hydro.UpdateUseIGR(useIGR);
	  } else{
		  hydro.UpdateUseIGR(useIGR);
		  hydro.UpdateUseVisc(visc);
	  }

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
      pmesh->NewNodes(x_gf, false);

      if (last_step || (ti % vis_steps) == 0)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         if (mem_usage)
         {
            mem = GetMaxRssMB();
            MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
            MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
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
               cout << ", mem: " << mmax << "/" << msum << " MB";
            }
            cout << endl;
         }

         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || visit || gfprint) { hydro.ComputeDensity(rho_gf); }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets
            if (problem != 0 && problem != 4)
            {
               hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                             "Density", Wx, Wy, Ww, Wh);
			   double rho_min = rho_gf.Min();
			   if(rho_min < 0.0){
	               mfem::out << "ERROR: negative density detected\n"
                   << "  time = " << t << "\n"
                   << "  rho in [" << rho_min << ", " << " " << "]\n";
	               MFEM_ABORT("Density became negative");
                }
            }
            Wx += offx;
            hydrodynamics::VisualizeField(vis_v, vishost, visport,
                                          v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                          "Specific Internal Energy",
                                          Wx, Wy, Ww, Wh);
            Wx += offx;
            if(useIGR && t > stallIGR){
               hydrodynamics::VisualizeField(vis_igr, vishost, visport, igr_gf,
                                             "IGR", Wx, Wy, Ww, Wh);
	           Wx += offx;
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
            std::ostringstream mesh_name, rho_name, v_name, e_name, igr_name;
            mesh_name << basename << "_" << ti << "_mesh";
            rho_name  << basename << "_" << ti << "_rho";
            v_name << basename << "_" << ti << "_v";
            e_name << basename << "_" << ti << "_e";
            igr_name  << basename << "_" << ti << "_igr";
            std::ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            pmesh->PrintAsOne(mesh_ofs);
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

            std::ofstream igr_ofs(igr_name.str().c_str());
            igr_ofs.precision(8);
            igr_gf.SaveAsOne(igr_ofs);
            igr_ofs.close();
         }
      }

      // Problems checks
      if (check)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         const double e_norm = sqrt(norm);
         MFEM_VERIFY(rs_levels==0 && rp_levels==0, "check: rs, rp");
         MFEM_VERIFY(order_v==2, "check: order_v");
         MFEM_VERIFY(order_e==1, "check: order_e");
         MFEM_VERIFY(ode_solver_type==4, "check: ode_solver_type");
         MFEM_VERIFY(t_final == 0.6, "check: t_final");
         MFEM_VERIFY(cfl==0.5, "check: cfl");
         MFEM_VERIFY(strncmp(mesh_file, "default", 7) == 0, "check: mesh_file");
         MFEM_VERIFY(dim==2 || dim==3, "check: dimension");
         Checks(ti, e_norm, checks);
      }
   }

   if (gfprint)
   {
      std::ostringstream mesh_name, rho_name, v_name, e_name, igr_name;
      const char *igr_suffix = "", *igr_folder = "WithIGR/";
      std::string problem_folder = "p" + std::to_string(problem) + "/", visc_suffix = "";
      if(!useIGR){
         igr_suffix = "_noigr";
         igr_folder = "WithoutIGR/";
      }
      if(visc){
        if(visc_const < 0){ 
          visc_suffix = "_LaghosVisc";
        } else{
          visc_suffix =  "_" + std::to_string(visc_const);
        }
      }
      if(TestPrint){ igr_folder = ""; problem_folder = ""; }

      mesh_name << basename << igr_folder << problem_folder << "Laghos_" << problem << "_" << rs_levels << "_" << ode_solver_type << "_" << t_final << igr_suffix << visc_suffix << "_mesh";
      rho_name  << basename << igr_folder << problem_folder << "Laghos_" << problem << "_" << rs_levels << "_" << ode_solver_type << "_" << t_final << igr_suffix << visc_suffix << "_rho";
      v_name << basename << igr_folder << problem_folder << "Laghos_" << problem << "_" << rs_levels << "_" << ode_solver_type << "_" << t_final << igr_suffix << visc_suffix << "_v";
      e_name << basename << igr_folder << problem_folder << "Laghos_" << problem << "_" << rs_levels << "_" << ode_solver_type << "_" << t_final << igr_suffix << visc_suffix << "_e";
      igr_name  << basename << igr_folder << problem_folder << "Laghos_" << problem << "_" << rs_levels << "_" << ode_solver_type << "_" << t_final << igr_suffix << visc_suffix << "_igr";
      std::ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->PrintAsOne(mesh_ofs);
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



   ////////////////////////////////////////////////////////////////////////

   if(write){

      cout << "Write \n";

      H1_FECollection H1FEClin(1, dim);
      ParFiniteElementSpace lin_fes(pmesh, &H1FEClin, pmesh->Dimension());

      // Project original solution to linear space
      GridFunction rho_lin(&lin_fes), v_lin(&lin_fes), e_lin(&lin_fes), igrp_lin(&lin_fes);
      rho_lin.ProjectGridFunction(rho_gf);
      v_lin.ProjectGridFunction(v_gf);
      e_lin.ProjectGridFunction(e_gf);
      igrp_lin.ProjectGridFunction(igr_gf);

      // Get true DOF values
      Vector rho_vals, v_vals, e_vals, igrp_vals;
      rho_lin.GetTrueDofs(rho_vals); // size = number of DOFs (vertices in 1D linear)
      v_lin.GetTrueDofs(v_vals);
      e_lin.GetTrueDofs(e_vals);
      igrp_lin.GetTrueDofs(igrp_vals);


      // Print values at mesh vertices in order
      //cout << "# x u\n";
      for (int i = 0; i < pmesh->GetNV(); i++) // NV = number of vertices
      {
          double xi = pmesh->GetVertex(i)[0]; // x-coordinate of vertex
          double rhoi = rho_vals[i];            // value at that vertex
          double vi = v_vals[i];
          double ei = e_vals[i];
          double igrpi = igrp_vals[i];
          //cout << " " << xi << ", " << rhoi << ", " << vi << ", " << ei << ", " << igrpi << ", \n";
      }

      std::ofstream outfile("../../ExactRiemannProblemSolver/data.csv"); // CSV is easy to read in Julia

      for (int i = 0; i < pmesh->GetNV(); i++) // NV = number of vertices
      {
         double xi = pmesh->GetVertex(i)[0]; // x-coordinate of vertex
         double rhoi = rho_vals[i];            // value at that vertex
         double vi = v_vals[i];
         double ei = e_vals[i];
         double igrpi = igrp_vals[i];    // value at that vertex
 
         // Write to file instead of console
         outfile << xi << ", " << rhoi << ", " << vi << ", " << ei << ", " << igrpi << ", \n";
      }

      // Close the file when done
      outfile.close();

   }


   ////////////////////////////////////////////////////////////


   MFEM_VERIFY(!check || checks == 2, "Check error!");
   
   //End timer
   auto end = chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;
   double elapsed_seconds = duration.count();
   cout << "Execution time: " << elapsed_seconds << " seconds." << endl; 
 
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
      mem = GetMaxRssMB();
      MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
      MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
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
         cout << "Maximum memory resident set size: "
              << mmax << "/" << msum << " MB" << endl;
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
   }

   // Free the used memory.
   delete ode_solver;
   delete pmesh;

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
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
      case 8: return 1.0;
	  case 9: return 1.0;
	  case 10: return 0.5*tanh(200*(0.5-x(0)))+.6;
     case 11: return 1.0;
     case 12: return (x(0) < 0.4) ? 1.0 : ((x(0) > 0.6) ? 0.1 : 1.0 - 4.5*(x(0) - 0.4));
     case 13: return 0.5*tanh(100*(0.5-x(0)))+.6;
     case 14: return ((x(0) < 0.4) ? 1.6*tanh(100*(0.32-x(0))) + 2.4 : 1 - 0.2*cos(50*(x(0)-0.4)));
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
         if (x(0) < 0.2 + (0.5 - length / 2) + 0.1 * length ) { 
                  v(0) = tanh(100*(x(0) - (0.5 - length / 2) - 0.06*length) / length ) + 1.0; 
         } else { v(0) = tanh(100.0*(0.35-x(0))) + 1.0; }
         break;
     }
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
	  //case 10: return 1/(0.5*tanh(20*(0.5-x(0)))+.6);
	  case 10: return 0.25;
     case 11: return 0.25;
     case 12: return 0.25*pow(( (x(0) < 0.4) ? 1.0 : ((x(0) > 0.6) ? 0.1 : 1.0 - 4.5*(x(0) - 0.4))),2.0);
     case 13: return 0.25*pow(0.5*tanh(100*(0.5-x(0)))+.6,2.0);
     case 14: return 2.5*( 4.5*tanh(100*(0.32-x(0))) + 5.5 )/ rho0(x)  ;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
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
   //printf("\033[33m%.15e\033[m\n",nrm);

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
      { chk++; MFEM_VERIFY(rerr(nrm, res, eps), "P"<<problem<<", #"<<i); }
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
         {{5, 1.339163718592566e+01}, { 28, 7.521073677397994e+00}},
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
         const int it = it_norms[dim-2][p][i][0];
         const double norm = it_norms[dim-2][p][i][1];
         check(p, it, norm);
      }
   }
}
