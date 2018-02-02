#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#define NK_PRIVATE
#include "nuklear.h"
#include "nuklear_glfw_gl3.h"

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

#define MAX_VERTEX_BUFFER (512*1024)
#define MAX_ELEMENT_BUFFER (128*1024)

#include <algorithm>
#include <type_traits>

#include <assert.h>
#include <float.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define UNUSED(x) x __attribute__ ((__unused__))

/**
 * array_size() - get the number of elements in array @arr.
 * @arr: array to be sized
 */
template<typename T, size_t N>
constexpr size_t
array_size(T (&)[N])
{
        return N;
}

struct environment {
        uint32_t finish_line_size;
        size_t grid_dim;
        uint32_t max_steps;
        uint32_t max_v;
        uint32_t num_actions;
};

template<size_t N>
struct p_equal {
        constexpr p_equal() : p{} {
                for (uint32_t i = 0;
                     i < N;
                     ++i) {
                        p[i] = 1.0/N;
                }
        }

        double p[N];
};

struct vec2 {
        uint32_t x;
        uint32_t y;
};

struct state {
        struct vec2 pos;
        struct vec2 v;
};

template<uint32_t max_steps>
struct histories {
        uint32_t actions[max_steps];
        int32_t rewards[max_steps];
        struct state states[max_steps];
};

static void
bind_buffer(uint32_t buffer,
            GLenum target,
            const void *data,
            uint32_t data_size_bytes)
{
        glBindBuffer(target, buffer);
        glBufferData(target, data_size_bytes, data, GL_STATIC_DRAW);
}

static uint32_t
create_compile_shader(const char **shader_src, GLenum shader_type)
{
        uint32_t shader = glCreateShader(shader_type);
        assert(shader != 0);

        glShaderSource(shader, 1, shader_src, NULL);
        glCompileShader(shader);

        int32_t success;
        char info_log[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (success == GL_FALSE) {
                glGetShaderInfoLog(shader,
                                   sizeof(info_log),
                                   NULL,
                                   info_log);
                printf("Error! Shader compilation failed\n%s\n", info_log);
                exit(EXIT_FAILURE);
        }

        return shader;
}

void error_callback(int32_t error, const char *description)
{
        fprintf(stderr, "Error: %d. %s\n", error, description);
}

static void
framebuffer_size_callback(GLFWwindow *UNUSED(window),
                          int32_t width,
                          int32_t height)
{
        glViewport(0, 0, width, height);
}

static void
failure(const char *msg_format_str, ...)
{
        va_list args;
        char buffer[64];

        va_start(args, msg_format_str);
        vsnprintf(buffer, sizeof(buffer), msg_format_str, args);
        fprintf(stderr, "%s", buffer);
        va_end(args);

        exit(EXIT_FAILURE);
}

/**
 * get_and_check_time_of_day() - Get time of day and store it in `tv`, checking
 * for errors.
 * @tv: Time of day output.
 */
static void
get_and_check_time_of_day(struct timeval *tv)
{
        int32_t status = gettimeofday(tv, NULL);
        assert(status == 0);
}

template<uint32_t num_actions>
static uint32_t
get_state_action_i(uint32_t state_i, uint32_t action_i)
{
        return state_i*num_actions + action_i;
}

template<typename T>
constexpr uint32_t
square(T val)
{
        return val*val;
}

template<uint32_t num_pos, uint32_t num_vel>
static uint32_t
get_state_i(struct state *st)
{
        uint32_t state_i = st->pos.y;
        state_i += num_pos*st->pos.x;
        state_i += square(num_pos)*st->v.y;
        state_i += num_vel*square(num_pos)*st->v.x;

        return state_i;
}

/**
 * get_seed_from_time_of_day() - Convenience function to return the 64-bit
 * micro-second part of the time of day.
 */
static uint64_t
get_seed_from_time_of_day(void)
{
        struct timeval seed;
        get_and_check_time_of_day(&seed);

        return seed.tv_usec;
}

static void
process_input(GLFWwindow *window)
{
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
}

static float
discrete_to_real(uint32_t discrete_point, uint32_t max_dim)
{
        return static_cast<float>(discrete_point)/max_dim - 0.5f;
}

template<size_t N>
static uint32_t
multinoulli(gsl_rng *rng)
{
        constexpr auto p_eq = p_equal<N>();
        uint32_t n[N];

        gsl_ran_multinomial(rng, N, 1, p_eq.p, n);
        for (uint32_t i = 0;
             i < N;
             ++i) {
                if (n[i] > 0)
                        return i;
        }

        return 0;
}

template<int32_t max_v>
static int32_t
clip_v(int32_t v, int32_t delta_v)
{
        return std::min(std::max(v + delta_v, 0), max_v);
}

template<size_t grid_dim>
static void
start_state(struct state *st, gsl_rng *rng)
{
        st->v = {.x = 0, .y = 0};
        st->pos = {.x = multinoulli<grid_dim>(rng), .y = 0};
}

template<uint32_t finish_line_size,
         size_t grid_dim,
         uint32_t max_steps,
         uint32_t max_v,
         uint32_t num_actions>
static uint32_t
run_episode(struct histories<max_steps> *hist,
            uint32_t *discrete_grid_x,
            gsl_rng *rng)
{
        struct state st;
        start_state<grid_dim>(&st, rng);

        uint32_t step_i;
        for (step_i = 0;
             step_i < (max_steps - 1);
             ++step_i) {
                /* NOTE(brendan): Rewards are -1 for each timestep. */
                hist->rewards[step_i] = -step_i;
                hist->states[step_i] = st;

                int32_t next_action = multinoulli<num_actions>(rng);
                hist->actions[step_i] = next_action;

                int32_t delta_vx = (next_action % 3) - 1;
                st.v.x = clip_v<max_v>(st.v.x, delta_vx);

                int32_t delta_vy = (next_action / 3) - 1;
                st.v.y = clip_v<max_v>(st.v.y, delta_vy);

                struct vec2 projected_pos = {.x = st.pos.x + st.v.x,
                                             .y = st.pos.y + st.v.y};

                /**
                 * NOTE(brendan): If the agent is projected to leave
                 * the racetrack by going above the track, then reset
                 * the agent to a random spot on the starting line.
                 *
                 * It is not possible for the agent to go below the
                 * track, because the agent's velocity is clipped to be
                 * non-negative in each direction.
                 */
                if (projected_pos.y >= grid_dim) {
                        start_state<grid_dim>(&st, rng);
                        continue;
                }

                uint32_t x0 = discrete_grid_x[2*projected_pos.y];
                uint32_t xf = discrete_grid_x[2*projected_pos.y + 1];
                /**
                 * NOTE(brendan): If the agent crosses the track
                 * boundary on the right at the finish line, the
                 * episode is done.  Otherwise, the agent is reset to
                 * the start line.
                 */
                if (projected_pos.x > xf) {
                        if (projected_pos.y >=
                            (grid_dim - finish_line_size)) {
                                uint32_t T = step_i + 1;
                                hist->rewards[T] = -T;
                                return T;
                        }

                        start_state<grid_dim>(&st, rng);
                        continue;
                }

                /**
                 * NOTE(brendan): The agent can also cross the track
                 * boundary on the left, by moving up into a region
                 * with no track.
                 */
                if (projected_pos.x < x0) {
                        start_state<grid_dim>(&st, rng);
                        continue;
                }

                st.pos = projected_pos;
        }

        /**
         * NOTE(brendan): Set R_T in the case of the episode overrunning
         * `max_steps`.
         */
        hist->rewards[step_i] = -step_i;
        return step_i;
}

int main(void)
{
        if (!glfwInit())
                failure("GLFW init failed!\n");

        glfwSetErrorCallback(error_callback);

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWwindow *window = glfwCreateWindow(800,
                                              600,
                                              "Bananas",
                                              NULL,
                                              NULL);
        if (window == NULL)
                failure("Could not open window.\n");

        glfwMakeContextCurrent(window);

        assert(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) != 0);

        int32_t width;
        int32_t height;
        glfwGetFramebufferSize(window, &width, &height);

        glViewport(0, 0, width, height);

        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        const char *vertex_shader_src =
                "#version 330 core\n"
                "layout (location = 0) in vec3 aPos;\n"
                ""
                "void main()\n"
                "{\n"
                "    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0f);\n"
                "}\n";
        const char *fragment_shader_src =
                "#version 330 core\n"
                "out vec4 FragColor;\n"
                ""
                "void main()\n"
                "{\n"
                "    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
                "}\n";

        uint32_t vertex_shader = create_compile_shader(&vertex_shader_src,
                                                       GL_VERTEX_SHADER);
        uint32_t fragment_shader = create_compile_shader(&fragment_shader_src,
                                                         GL_FRAGMENT_SHADER);

        uint32_t shader_program = glCreateProgram();
        assert(shader_program != 0);

        glAttachShader(shader_program, vertex_shader);
        glAttachShader(shader_program, fragment_shader);
        glLinkProgram(shader_program);

        int32_t success;
        glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
        if (success == GL_FALSE) {
                char info_log[512];
                glGetProgramInfoLog(shader_program,
                                    sizeof(info_log),
                                    NULL,
                                    info_log);
                printf("Shader program linking failed!\n%s\n", info_log);
                exit(EXIT_FAILURE);
        }

        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        const gsl_rng_type *rng_type = gsl_rng_taus;
        gsl_rng *rng = gsl_rng_alloc(rng_type);
        assert(rng != NULL);

        gsl_rng_set(rng, get_seed_from_time_of_day());

        /**
         *
         *             max y +----> +-----------------------+-+
         *                      +---+                       | |
         *                      |                           | |
         *                    +-+                           | | finish
         *                    |                             | |
         *                    |                             | |
         *                    |                       +-----+-+
         *                    +---+                 +-+       ^
         *                        |               +-+         |
         *                        |            +--+           |
         *                      +-+  min width |              |
         *                      |             ++              +
         *      min x      +----+             |               max x
         *      +        +-+                +-+
         *      |    +---+                  |
         *      v +--+                      |
         *      +-+                        ++
         *      |--------------------------|
         *      +--------------------------+<----+ min y
         *               begin
         */
        constexpr struct environment env = {.finish_line_size = 8,
                                            .grid_dim = 32,
                                            .max_steps = 65536,
                                            .max_v = 4,
                                            .num_actions = 9};

        constexpr uint32_t min_width = env.grid_dim/4;
        uint32_t discrete_grid_x[2*env.grid_dim];
        float grid_vertices[3*2*env.grid_dim];
        uint32_t x0 = 0;
        uint32_t xf = env.grid_dim/2;
        for (uint32_t line_i = 0;
             line_i < env.grid_dim;
             ++line_i) {
                if (line_i == (env.grid_dim - env.finish_line_size))
                        xf = env.grid_dim - 1;

                if (gsl_ran_flat(rng, 0, 1) < 0.75) {
                        /* NOTE(brendan): change line width. */
                        if ((line_i < env.grid_dim/2) && ((xf - x0) > min_width))
                                x0 += 1;
                        else if ((line_i >= env.grid_dim/2) &&
                                 (xf < (env.grid_dim - 1)))
                                xf += 1;
                }

                discrete_grid_x[2*line_i] = x0;
                discrete_grid_x[2*line_i + 1] = xf;

                const uint32_t offset = 3*2*line_i;
                const float y = discrete_to_real(line_i, env.grid_dim);
                grid_vertices[offset + 0] = discrete_to_real(x0, env.grid_dim);
                grid_vertices[offset + 1] = y;
                grid_vertices[offset + 2] = 0.0f;

                grid_vertices[offset + 3] = discrete_to_real(xf, env.grid_dim);
                grid_vertices[offset + 4] = y;
                grid_vertices[offset + 5] = 0.0f;
        }

        constexpr size_t num_points = array_size(grid_vertices)/3;
        uint32_t indices[num_points];
        for (uint32_t i = 0;
             i < num_points;
             ++i) {
                indices[i] = i;
        }

        /**
         * NOTE(brendan): At each timestep, the car is at one of a discrete set
         * of grid cells, with 0 <= x < grid_dim, and 0 <= y < grid_dim.
         *
         * The velocity is also a discrete number of cells moved vertically and
         * horizontally per timestep, with 0 <= vx < 5 and 0 <= vy < 5.
         *
         * Each velocity component vx and vy can be changed -1, +1 or 0 in one
         * timestep. So, there are nine actions.
         *
         * Each episode begins in a randomly selected start state with y == 0,
         * and x a randomly chosen grid cell on the starting line. In the start
         * state, vx == vy == 0.
         *
         * The episode ends when the car crosses (reaches?) the finish line.
         *
         * Rewards are -1 for each timestep.
         *
         * If the car hits the track boundary, it is moved back to a random
         * position on the start line, and returned to vx == vy == 0.
         *
         * With probability 0.1 at each timestep, the velocity increments are
         * both zero regardless of the action.
         */

        /**
         * Off-policy MC control algorithm, from Sutton 5.7.
         *
         * 0 Initialize, for all s in S, a in A(s):
         * 1     Q(s, a) <- arbitrary
         * 2     C(s, a) <- 0
         * 3     \pi(s) <- argmax_a Q(s, a)  (with ties broken consistently)
         *
         * 4 Repeat forever:
         * 5     b <- any soft policy
         * 6     Generate an episode using b:
         * 7         S_0, A_0, R_1, ..., S_{T - 1}, A_{T - 1}, R_T, S_T
         * 8     G <- 0
         * 9     W <- 1
         * 10    For t = T - 1, T - 2, ... down to 0:
         * 11        G <- \gamma*G + R_{t + 1}
         * 12        C(S_t, A_t) <- C(S_t, A_t) + W
         * 13        Q(S_t, A_t) <- Q(S_t, A_t) +
         *                          W/C(S_t, A_t)*(G - Q(S_t, A_t))
         * 14        \pi(S_t) <- argmax_a Q(S_t, a)
         * 15        If A_t != \pi(S_t) then exit for loop
         * 16        W <- W*1/b(A_t | S_t)
         */

        constexpr uint32_t num_actions = 9;
        constexpr uint32_t num_vel = env.max_v + 1;
        constexpr uint32_t num_states = (square(env.grid_dim) *
                                         square(num_vel));
        constexpr uint32_t max_state_actions = num_states*num_actions;

        double q[max_state_actions];
        uint32_t c[max_state_actions];
        uint32_t pi[num_states];
        for (uint32_t state_i = 0;
             state_i < num_states;
             ++state_i) {
                for (uint32_t action_i = 0;
                     action_i < num_actions;
                     ++action_i) {
                        uint32_t state_action_i =
                                get_state_action_i<num_actions>(state_i,
                                                                action_i);
                        q[state_action_i] = 0;
                        c[state_action_i] = 0;
                }

                pi[state_i] = 0;
        }

        for (;;) {
                /* 6 Generate an episode using b. */
                struct histories<env.max_steps> hist;

                /* NOTE(brendan): `epi_steps` == T */
                uint32_t epi_steps =
                        run_episode<env.finish_line_size,
                                    env.grid_dim,
                                    env.max_steps,
                                    env.max_v,
                                    env.num_actions>(&hist,
                                                     discrete_grid_x,
                                                     rng);
                assert(epi_steps < env.max_steps);

                constexpr double gamma = 0.99;
                double return_G = 0;
                double weight_W = 1.0;
                for (int32_t step_i = epi_steps - 1;
                     step_i >= 0;
                     --step_i) {
                        /* 11 G <- \gamma*G + R_{t + 1} */
                        return_G = gamma*return_G + hist.rewards[step_i + 1];

                        /* 12 C(S_t, A_t) <- C(S_t, A_t) + W */
                        uint32_t action_i = hist.actions[step_i];
                        struct state *st_i = hist.states + step_i;
                        uint32_t state_i =
                                get_state_i<env.grid_dim, num_vel>(st_i);
                        uint32_t sa_i =
                                get_state_action_i<num_actions>(state_i,
                                                                action_i);
                        c[sa_i] += weight_W;

                        /**
                         * 13 Q(S_t, A_t) <- Q(S_t, A_t) +
                         *                   W/C(S_t, A_t)*(G - Q(S_t, A_t))
                         */
                        q[sa_i] += weight_W/c[sa_i]*(return_G - q[sa_i]);

                        /* 14 \pi(S_t) <- argmax_a Q(S_t, a) */
                        uint32_t best_action;
                        double max_value = -DBL_MAX;
                        for (uint32_t possible_action = 0;
                             possible_action < num_actions;
                             ++possible_action) {
                                uint32_t possible_sa_i =
                                        get_state_action_i<num_actions>(state_i,
                                                                        possible_action);
                                if (q[possible_sa_i] > max_value) {
                                        max_value = q[possible_sa_i];
                                        best_action = possible_action;
                                }
                        }

                        /* 15 If A_t != \pi(S_t) then exit for loop */
                        if (action_i != best_action)
                                break;

                        /**
                         * 16 W <- W*1/b(A_t | S_t)
                         *
                         * NOTE(brendan): Since the behaviour policy is
                         * uniformly distributed over all possible actions, the
                         * probability of any given action is `1/num_actions`
                         * regardless of state.
                         */
                        weight_W /= num_actions;
                }
        }

        /**
         * NOTE(brendan): Drawing is below here.
         */
        uint32_t EBO;
        glGenBuffers(1, &EBO);

        uint32_t VBO;
        glGenBuffers(1, &VBO);

        uint32_t VAO;
        glGenVertexArrays(1, &VAO);

        glBindVertexArray(VAO);
        bind_buffer(EBO, GL_ELEMENT_ARRAY_BUFFER, indices, sizeof(indices));
        bind_buffer(VBO,
                    GL_ARRAY_BUFFER,
                    grid_vertices,
                    sizeof(grid_vertices));

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), NULL);
        glEnableVertexAttribArray(0);

        struct nk_context *ctx = nk_glfw3_init(window,
                                               NK_GLFW3_INSTALL_CALLBACKS);

        struct nk_font_atlas *atlas;
        nk_glfw3_font_stash_begin(&atlas);
        nk_glfw3_font_stash_end();

        glfwSwapInterval(1);

        struct nk_color background = nk_rgb(28,48,62);
        while (!glfwWindowShouldClose(window)) {
                nk_glfw3_new_frame();

                process_input(window);

                enum {FILL, LINE};
                static int line_op = FILL;
                /* GUI */
                if (nk_begin(ctx, "Demo", nk_rect(50, 50, 230, 250),
                             NK_WINDOW_BORDER|NK_WINDOW_MOVABLE|NK_WINDOW_SCALABLE|
                             NK_WINDOW_MINIMIZABLE|NK_WINDOW_TITLE)) {
                        nk_layout_row_dynamic(ctx, 30, 2);
                        if (nk_option_label(ctx, "fill", line_op == FILL))
                                line_op = FILL;
                        if (nk_option_label(ctx, "line", line_op == LINE))
                                line_op = LINE;
                }
                nk_end(ctx);

                /* Draw */
                float bg[4];
                nk_color_fv(bg, background);
                glfwGetWindowSize(window, &width, &height);
                glViewport(0, 0, width, height);
                glClear(GL_COLOR_BUFFER_BIT);
                glClearColor(bg[0], bg[1], bg[2], bg[3]);

                if (line_op == LINE)
                        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

                glUseProgram(shader_program);
                glBindVertexArray(VAO);
                glDrawElements(GL_LINES, num_points, GL_UNSIGNED_INT, NULL);

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

                /* IMPORTANT: `nk_glfw_render` modifies some global OpenGL state
                 * with blending, scissor, face culling, depth test and viewport and
                 * defaults everything back into a default state.
                 * Make sure to either a.) save and restore or b.) reset your own state after
                 * rendering the UI. */
                nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);

                glfwSwapBuffers(window);
                glfwPollEvents();
        }

        nk_glfw3_shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        exit(EXIT_SUCCESS);
}
