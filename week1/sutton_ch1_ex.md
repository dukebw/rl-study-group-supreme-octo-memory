1.1: Self play.

        - If both sides learn, the model may not have a chance to learn a
          single value function to beat the opponent, as the opponent will also
          be constantly improving.

        - The random exploratory moves of the opponent should allow for enough
          variation for the model to learn.


1.2: Symmetries.

        - States of the symmetries would have to be mapped onto each other, and
          the corresponding actions would also be joined.

        - The learning process would be improved by reducing the search space
          of actions. It would be quicker to learn the true distribution of the
          value function.

        - If the opponent did not take advantage of symmetries, then the
          opponent's value function would look different for positions that are
          otherwise equivalent. Therefore, symmetrically equivalent positions
          should not necessarily have the same value, as those positions' value
          depends on the play of the opponent, which may not share the same
          symmetry.


1.3: Greedy play.

        - The only difference with the described RL algorithm seems to be the
          lack of exploratory moves. So the disadvantage of always playing
          greedily would be in potentially not exploring more optimal parts of
          the search space due to those parts' current perceived
          sub-optimality.


1.4: Learning from exploration.

        - If exploration rate is delta, the value function would converge to
          (1 - delta)*P_opt + delta*P_avg, i.e. the interpolation between the
          optimal probabilities and the average probabilities of winning when
          taking a step. This is because for delta proportion of total actions,
          the value function for each state is updated with the value function
          resulting from a randomly chosen action.

        - The optimal probabilities should be better to learn, since by
          definition of being optimal they will be better than average.

1.5: Other improvements.

        - Cache moves in a game that results in losing so that more of the
          search space starting from a set of initial moves can be explored at
          once. This could do a better job of propagating back the true value
          of an initial move.
