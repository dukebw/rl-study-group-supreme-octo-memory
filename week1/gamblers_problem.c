#include "glib.h"
#include "gsl/gsl_math.h"
#include "plplot/plplot.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DECAY_FACTOR 1.0
#define PROB_H (0.49)
#define WIN_CAPITAL 100
#define NUM_STATES (WIN_CAPITAL + 1)

static void
print_value(double *value)
{
        for (uint32_t capital = 0;
             capital < NUM_STATES;
             ++capital) {
                printf("%.3f ", value[capital]);
                if ((capital > 0) && (capital % 10) == 0)
                        printf("\n");
        }
        printf("\n");
}

static double
get_state_value(uint32_t *best_policy, uint32_t capital, double *value)
{
        uint32_t max_stake = MIN(capital,
                                 WIN_CAPITAL - capital);
        double best_policy_val = GSL_NEGINF;
        for (uint32_t candidate_policy = 1;
             candidate_policy <= max_stake;
             ++candidate_policy) {
                uint32_t heads_state = capital + candidate_policy;
                uint32_t tails_state = capital - candidate_policy;

                double policy_val = (PROB_H*value[heads_state] +
                                     (1.0 - PROB_H)*value[tails_state]);

                if (policy_val > best_policy_val) {
                        best_policy_val = policy_val;

                        if (best_policy != NULL)
                                *best_policy = candidate_policy;
                }
        }

        return best_policy_val;
}

int main(int argc, const char **argv)
{
        double value[NUM_STATES];

        for (uint32_t value_i = 0;
             value_i < (G_N_ELEMENTS(value) - 1);
             ++value_i) {
                value[value_i] = 0.0;
        }
        value[G_N_ELEMENTS(value) - 1] = 1.0;

        double delta;
        do {
                delta = 0.0;

                // for each s in S
                for (uint32_t capital = 1;
                     capital < WIN_CAPITAL;
                     ++capital) {
                        // v <- V(s)
                        double prev_value = value[capital];

                        // V(s) <-
                        // max_a \sum_{s', r} p(s',r|s,a)*[r + \gamma*V(s')]
                        value[capital] = get_state_value(NULL, capital, value);

                        // delta <- max(delta, |v - V(s)|)
                        double val_diff = prev_value - value[capital];
                        delta = MAX(delta, sqrt(val_diff*val_diff));
                }

                print_value(value);
        } while(delta > 0.00001);

        uint32_t policy[NUM_STATES];
        for (uint32_t capital = 1;
             capital < WIN_CAPITAL;
             ++capital) {
                get_state_value(policy + capital, capital, value);
        }

        plparseopts(&argc, argv, PL_PARSE_FULL);

        plinit();

        plenv(0, WIN_CAPITAL, 0, 55, 0, 0);

        double capital_range[NUM_STATES];
        double policy_plot[NUM_STATES];
        for (uint32_t capital = 0;
             capital < NUM_STATES;
             ++capital) {
                capital_range[capital] = capital;
                policy_plot[capital] = policy[capital];
        }
        policy_plot[0] = 0.0;

        plbin(NUM_STATES, capital_range, policy_plot, 0);

        plenv(0, WIN_CAPITAL, 0, 1.0, 0, 0);

        plline(NUM_STATES, capital_range, value);

        plend();

        exit(EXIT_SUCCESS);
}
