#include <glib.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <omp.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DECAY_FACTOR 0.9
#define MAX_CARS 16
#define NUM_PER_LOT_STATES (MAX_CARS + 1)
#define MAX_MOVE 5
#define REWARD_MOVE (-2.0)
#define REWARD_RENT (10.0)
#define REWARD_EXTRA_LOT (-4.0)
#define EXTRA_LOT_NUM_CARS 10

struct car_lot {
        const double l_requests;
        const double l_returns;
        double req_probs[MAX_CARS + 1];
        double return_probs[MAX_CARS + 1];
};

static bool
is_valid_transfer(uint32_t first_lot_num_cars,
                  uint32_t second_lot_num_cars,
                  int32_t transfer)
{
        uint32_t from_lot_num_cars;
        uint32_t to_lot_num_cars;
        if (transfer > 0) {
                from_lot_num_cars = first_lot_num_cars;
                to_lot_num_cars = second_lot_num_cars;
        } else {
                from_lot_num_cars = second_lot_num_cars;
                to_lot_num_cars = first_lot_num_cars;
        }

        uint32_t num_moved = abs(transfer);

        return ((from_lot_num_cars >= num_moved) &&
                ((to_lot_num_cars + num_moved) <= MAX_CARS));
}

static uint32_t
get_state(uint32_t first_lot_num_cars, uint32_t second_lot_num_cars)
{
        uint32_t state = (NUM_PER_LOT_STATES*first_lot_num_cars + second_lot_num_cars);
        assert(state < NUM_PER_LOT_STATES*NUM_PER_LOT_STATES);

        return state;
}

static uint32_t
adjust_num_cars(uint32_t *lot_num_cars, uint32_t num_reqs, uint32_t num_returns)
{
        uint32_t num_satisfied_reqs = MIN(num_reqs, *lot_num_cars);
        *lot_num_cars -= num_satisfied_reqs;
        *lot_num_cars += num_returns;
        *lot_num_cars = MIN(*lot_num_cars, MAX_CARS);

        return num_satisfied_reqs;
}

static double
get_state_action_value(struct car_lot *first_lot,
                       struct car_lot *second_lot,
                       uint32_t first_lot_num_cars,
                       uint32_t second_lot_num_cars,
                       int32_t policy,
                       double *prev_value)
{
        double value = 0.0;
        double reward_move = REWARD_MOVE*abs(policy);

        uint32_t first_lot_num_cars_start = first_lot_num_cars - policy;
        uint32_t second_lot_num_cars_start = second_lot_num_cars + policy;
        double reward_extra_lot = 0.0;
        // if more than ten cars are stored at either location, a
        // REWARD_EXTRA_LOT cost is incurred.
        if ((first_lot_num_cars_start > EXTRA_LOT_NUM_CARS) ||
            (second_lot_num_cars_start > EXTRA_LOT_NUM_CARS))
                reward_extra_lot = REWARD_EXTRA_LOT;

        // iterate over all states and rewards, cutting off the number of
        // requests and returns at MAX_CARS, as more than MAX_CARS requests
        // will never be satisfied or stored.
#pragma omp parallel for reduction(+:value)
        for (uint32_t first_lot_num_reqs = 0;
             first_lot_num_reqs <= MAX_CARS;
             ++first_lot_num_reqs) {
                for (uint32_t first_lot_num_returns = 0;
                     first_lot_num_returns <= MAX_CARS;
                     ++first_lot_num_returns) {
                        for (uint32_t second_lot_num_reqs = 0;
                             second_lot_num_reqs <= MAX_CARS;
                             ++second_lot_num_reqs) {
                                for (uint32_t second_lot_num_returns = 0;
                                     second_lot_num_returns <= MAX_CARS;
                                     ++second_lot_num_returns) {
                                        uint32_t first_lot_num_cars = first_lot_num_cars_start;
                                        uint32_t second_lot_num_cars = second_lot_num_cars_start;

                                        uint32_t first_lot_satisfied_reqs =
                                                adjust_num_cars(&first_lot_num_cars,
                                                                first_lot_num_reqs,
                                                                first_lot_num_returns);
                                        uint32_t second_lot_satisfied_reqs =
                                                adjust_num_cars(&second_lot_num_cars,
                                                                second_lot_num_reqs,
                                                                second_lot_num_returns);

                                        uint32_t new_state =
                                                get_state(first_lot_num_cars,
                                                          second_lot_num_cars);
                                        double prob = (first_lot->req_probs[first_lot_num_reqs]*
                                                       first_lot->return_probs[first_lot_num_returns]*
                                                       second_lot->req_probs[second_lot_num_reqs]*
                                                       second_lot->return_probs[second_lot_num_returns]);
                                        value += prob*(reward_move +
                                                       reward_extra_lot +
                                                       REWARD_RENT*(first_lot_satisfied_reqs +
                                                                    second_lot_satisfied_reqs) +
                                                       DECAY_FACTOR*prev_value[new_state]);
                                }
                        }
                }
        }

        return value;
}

static double
update_value_iter(double *value,
                  int32_t *policy,
                  struct car_lot *first_lot,
                  struct car_lot *second_lot)
{
        double prev_value[NUM_PER_LOT_STATES*NUM_PER_LOT_STATES];
        memcpy(prev_value, value, sizeof(prev_value));

        double delta = 0.0;
        // for each s in S
        for (uint32_t first_lot_i = 0;
             first_lot_i < NUM_PER_LOT_STATES;
             ++first_lot_i) {
                for (uint32_t second_lot_i = 0;
                     second_lot_i < NUM_PER_LOT_STATES;
                     ++second_lot_i) {
                        // NOTE(brendan): it is assumed that deciding how many
                        // cars to transfer happens at the _beginning_ of the
                        // night, and therefore returned cars can't be
                        // transferred.
                        uint32_t state = get_state(first_lot_i, second_lot_i);
                        assert(is_valid_transfer(first_lot_i,
                                                 second_lot_i,
                                                 policy[state]));

                        value[state] = get_state_action_value(first_lot,
                                                              second_lot,
                                                              first_lot_i,
                                                              second_lot_i,
                                                              policy[state],
                                                              prev_value);

                        double val_diff = value[state] - prev_value[state];
                        delta = MAX(val_diff*val_diff, delta);
                }
        }

        return delta;
}

#define PRINT_FN(Name) void Name(void *grid, uint32_t state)
typedef PRINT_FN(print_fn);

static void
print_value_elem(void *grid, uint32_t state)
{
        double *value = grid;
        printf("%.1f  ", value[state]);
}

static void
print_policy_elem(void *grid, uint32_t state)
{
        int32_t *policy = grid;
        printf("%d  ", policy[state]);
}

static void
print_grid(char *name, void *grid, print_fn *print_grid_element)
{
        printf("%s\n", name);

        for (uint32_t first_lot_i = 0;
             first_lot_i < NUM_PER_LOT_STATES;
             ++first_lot_i) {
                for (uint32_t second_lot_i = 0;
                     second_lot_i < NUM_PER_LOT_STATES;
                     ++second_lot_i) {
                        uint32_t state = get_state(first_lot_i, second_lot_i);
                        print_grid_element(grid, state);
                }
                printf("\n");
        }
        printf("\n");
}

// Evaluate
// +10 dollars per car rented out.
// REWARD_MOVE dollars per car moved.
// Max. 5 cars moved per night.
static void
evaluate_policy(double *value,
                int32_t *policy,
                struct car_lot *first_lot,
                struct car_lot *second_lot)
{
        double delta;
        do {
                delta = update_value_iter(value,
                                          policy,
                                          first_lot,
                                          second_lot);
        } while (delta >= 0.0001);
}

static bool
improve_policy_iter(struct car_lot *first_lot,
                    struct car_lot *second_lot,
                    uint32_t first_lot_i,
                    uint32_t second_lot_i,
                    double *value,
                    int32_t *policy)
{
        bool is_policy_stable = true;
        uint32_t state = get_state(first_lot_i, second_lot_i);
        // old action <- pi(s)
        int32_t best_policy = policy[state];
        double best_policy_val = GSL_NEGINF;
        // pi(s) <- argmax_a p(s',r|s,a)[r + \gamma*V(s')]
        // One extra car can be shuttled from the first to second location.
        for (int32_t candidate_policy = -(MAX_MOVE + 1);
             candidate_policy <= MAX_MOVE;
             ++candidate_policy) {
                if (is_valid_transfer(first_lot_i,
                                      second_lot_i,
                                      candidate_policy)) {
                        double policy_val =
                                get_state_action_value(first_lot,
                                                       second_lot,
                                                       first_lot_i,
                                                       second_lot_i,
                                                       candidate_policy,
                                                       value);
                        if (policy_val > best_policy_val) {
                                best_policy_val = policy_val;
                                best_policy = candidate_policy;
                        }
                }
        }

        if (best_policy != policy[state]) {
                is_policy_stable = false;
                policy[state] = best_policy;
        }

        return is_policy_stable;
}

static bool
improve_policy(struct car_lot *first_lot,
               struct car_lot *second_lot,
               int32_t *policy,
               double *value)
{
        bool is_policy_stable = true;
        // for each s in S
        for (uint32_t first_lot_i = 0;
             first_lot_i < NUM_PER_LOT_STATES;
             ++first_lot_i) {
                for (uint32_t second_lot_i = 0;
                     second_lot_i < NUM_PER_LOT_STATES;
                     ++second_lot_i) {
                        is_policy_stable = (improve_policy_iter(first_lot,
                                                                second_lot,
                                                                first_lot_i,
                                                                second_lot_i,
                                                                value,
                                                                policy) &&
                                            is_policy_stable);
                }
        }

        return is_policy_stable;
}

static void
set_tail_prob(double *probs, uint32_t num_bins)
{
        double prob_cumul = 0.0;
        for (uint32_t prob_i = 0;
             prob_i < (num_bins - 1);
             ++prob_i) {
                prob_cumul += probs[prob_i];
        }

        assert(prob_cumul <= 1.0);
        probs[num_bins - 1] = 1.0 - prob_cumul;
}

int main(void)
{
        struct car_lot first_lot = {
                .l_requests = 3.0,
                .l_returns = 3.0,
        };

        struct car_lot second_lot = {
                .l_requests = 4.0,
                .l_returns = 2.0,
        };

        for (uint32_t k = 0;
             k <= MAX_CARS;
             ++k) {
                first_lot.req_probs[k] =
                        gsl_ran_poisson_pdf(k, first_lot.l_requests);
                first_lot.return_probs[k] =
                        gsl_ran_poisson_pdf(k, first_lot.l_returns);
                second_lot.req_probs[k] =
                        gsl_ran_poisson_pdf(k, second_lot.l_requests);
                second_lot.return_probs[k] =
                        gsl_ran_poisson_pdf(k, second_lot.l_returns);
        }

        set_tail_prob(first_lot.req_probs, G_N_ELEMENTS(first_lot.req_probs));
        set_tail_prob(first_lot.return_probs,
                      G_N_ELEMENTS(first_lot.return_probs));
        set_tail_prob(second_lot.req_probs,
                      G_N_ELEMENTS(second_lot.req_probs));
        set_tail_prob(second_lot.return_probs,
                      G_N_ELEMENTS(second_lot.return_probs));

        int32_t policy[NUM_PER_LOT_STATES*NUM_PER_LOT_STATES];
        memset(policy, 0, sizeof(policy));

        double value[NUM_PER_LOT_STATES*NUM_PER_LOT_STATES];
        memset(value, 0, sizeof(value));

        // Policy iteration algorithm from section 4.3 of Sutton.
        bool is_policy_stable;
        do {
                evaluate_policy(value, policy, &first_lot, &second_lot);

                // Improve
                is_policy_stable = improve_policy(&first_lot,
                                                  &second_lot,
                                                  policy,
                                                  value);

                print_grid("policy", policy, print_policy_elem);
                print_grid("value", value, print_value_elem);
        } while (!is_policy_stable);

        return EXIT_SUCCESS;
}
