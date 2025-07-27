add_subdirectory(OdeSolver)

add_library(Render Shader.cpp Particles.cpp Renderer.cpp )
target_link_libraries(Render PRIVATE glad glfw glm Solvers)

add_executable(main main.cpp)
target_link_libraries(main  PRIVATE Render glad)