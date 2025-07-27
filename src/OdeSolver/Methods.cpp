#include "OdeSolver/Methods.hpp"

std::unordered_map<SolverKey, SolverLauncher, SolverKeyHash> kernelMap{
    std::pair<const SolverKey, SolverLauncher>(SolverKey{SolverMethod::EULER, SolverFlags::FLAG_DEFAULT}, launch_euler_pm),
    std::pair<const SolverKey, SolverLauncher>(SolverKey{SolverMethod::EULER, SolverFlags::FLAG_P2},      launch_euler_pm_p2)
};
