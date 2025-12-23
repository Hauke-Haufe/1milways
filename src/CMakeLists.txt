add_subdirectory(OdeSolver)
add_subdirectory(ICGenerator)

add_library(Render Shader.cpp ParticleSimulation.cpp Renderer.cpp ParticleRenderer.cpp)
target_link_libraries(Render PRIVATE glad glfw glm Solvers)

add_executable(main main.cpp)
target_link_libraries(main  PRIVATE Render glad)