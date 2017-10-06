#include <glib.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CARS 20
#define DECAY_FACTOR 0.9

struct car_lot {
        const double l_requests;
        const double l_returns;
        uint32_t num_cars;
};

static void
check_valid_transfer(struct car_lot *from_lot,
                     struct car_lot *to_lot,
                     uint32_t num_moved)
{
        assert((from_lot->num_cars >= num_moved) &&
               ((to_lot->num_cars + num_moved) <= MAX_CARS));
}

static uint32_t
car_lot_env(struct car_lot *lot, gsl_rng *rng)
{
        uint32_t lot_num_requests = gsl_ran_poisson(rng, lot->l_requests);
        uint32_t lot_num_returns = gsl_ran_poisson(rng, lot->l_returns);

        uint32_t num_satisfied_reqs = MIN(lot_num_requests, lot->num_cars);
        lot->num_cars -= num_satisfied_reqs;
        lot->num_cars += lot_num_returns;

        return num_satisfied_reqs;
}

static uint32_t
get_state(struct car_lot *first_lot, struct car_lot *second_lot)
{
        return (first_lot->num_cars + MAX_CARS*second_lot->num_cars);
}

static double
update_value_iter(double *value,
                  int32_t *policy,
                  struct car_lot *first_lot,
                  struct car_lot *second_lot,
                  gsl_rng *rng)
{
        double prev_value[MAX_CARS*MAX_CARS];
        memcpy(prev_value, value, sizeof(prev_value));

        for (uint32_t first_lot_i = 0;
             first_lot_i < MAX_CARS;
             ++first_lot_i) {
                for (uint32_t second_lot_i = 0;
                     second_lot_i < MAX_CARS;
                     ++second_lot_i) {
                        first_lot->num_cars = first_lot_i;
                        second_lot->num_cars = second_lot_i;

                        // NOTE(brendan): it is assumed that deciding how many
                        // cars to transfer happens at the _beginning_ of the
                        // night, and therefore returned cars can't be
                        // transferred.
                        uint32_t state = get_state(first_lot, second_lot);

                        uint32_t num_moved = abs(policy[state]);
                        if (policy[state] > 0)
                                check_valid_transfer(first_lot, second_lot, num_moved);
                        else if (policy[state] < 0)
                                check_valid_transfer(second_lot, first_lot, num_moved);

                        first_lot->num_cars -= policy[state];
                        second_lot->num_cars += policy[state];

                        int32_t reward = -2*num_moved;

                        reward += 10*(car_lot_env(first_lot, rng) +
                                      car_lot_env(second_lot, rng));

                        uint32_t new_state = get_state(first_lot, second_lot);
                        value[state] = reward + 0.9*prev_value[new_state];
                }
        }

        double delta = 0.0;
        for (first_lot->num_cars = 0;
             first_lot->num_cars < MAX_CARS;
             ++first_lot->num_cars) {
                for (second_lot->num_cars = 0;
                     second_lot->num_cars < MAX_CARS;
                     ++second_lot->num_cars) {
                        uint32_t state = get_state(first_lot,
                                                   second_lot);
                        double val_diff = value[state] - prev_value[state];
                        delta += val_diff*val_diff;
                }
        }

        return delta;
}

// Evaluate
// +10 dollars per car rented out.
// -2 dollars per car moved.
// Max. 5 cars moved per night.
static void
evaluate_policy(double *value,
                int32_t *policy,
                struct car_lot *first_lot,
                struct car_lot *second_lot,
                gsl_rng *rng)
{
        double delta;
        do {
                delta = update_value_iter(value,
                                          policy,
                                          first_lot,
                                          second_lot,
                                          rng);

        } while (delta >= 0.001);
}

int main(void)
{
        gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
        assert(rng != NULL);

        struct car_lot first_lot = {
                .l_requests = 3.0,
                .l_returns = 3.0,
                .num_cars = 10,
        };

        struct car_lot second_lot = {
                .l_requests = 4.0,
                .l_returns = 2.0,
                .num_cars = 10,
        };

        int32_t policy[MAX_CARS*MAX_CARS];
        memset(policy, 0, sizeof(policy));

        double value[MAX_CARS*MAX_CARS];
        memset(value, 0, sizeof(value));

        bool is_policy_stable;
        do {
                evaluate_policy(value, policy, &first_lot, &second_lot, rng);

                // Improve
                // TODO(brendan): assign to policy[state] the action that
                // maximizes [r + gamma*V(s')].
                is_policy_stable = true;
        } while (!is_policy_stable);

        gsl_rng_free(rng);

        return EXIT_SUCCESS;
}
