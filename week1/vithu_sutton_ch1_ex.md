# Chapter 1 Exercises

1.1: Suppose, instead of playing against a random opponent, the reinforcement learning algorithm described above played against itself, with both sides learning. What do you think would happen in this case? Would it learn a different policy for selecting moves?

	- RL algorithm would learn to play against itself
	- The algorithm would learn the minmax way of playing the game because the algorithm consisted of picking a greedy move and this is exactly what the opponent would do too
	- The difference between the minmax algorithm here and this RL algorithm is the presence of random exploratory moves, but this wouldn't disrupt the policy learnt, seeing as the exploratory moves don't contribute to any learning

1.2 Many tic-tac-toe positions appear different but are really the same because of symmetries. How might we amend the reinforcement learning algorithm described above to take advantage of this? In what ways would this improve it? Now think again. Suppose the opponent did not take advantage of symmetries. In that case, should we? Is it true, then, that symmetrically equivalent positions should necessarily have the same 
value?

	- It's possible to use 4 axis of symmetry to basically fold the board down to a quarter of the size
	- Would cause a significant increase in speed and reduction in memory required 
	- If the opponent doesn't take advantage of this, then it'd result in worse overall performance
	- Example: the opponent always played correct except for 1 corner, then you never really took advantage the information given by the symmetries
	- In that case, symmetrically equivalent positions don't always hold the same value in a multi-player gmae

1.3 Suppose the reinforcement learning player was greedy, that is, it always played the move that brought it to the position that it rated the best. Would it learn to play better, or worse, than a non-greedy player? What problems might occur?

	- Generally, it'd play worse
	- Chances are slim, when correct action for a situation in the long run is the first one that returns a positive reward
	- Especially, if there are large number of actions available
	- Would be rather difficult/unable to adapt to opponents that slowly changed behaviour over time

1.4 Suppose learning updates occurred after all moves, including exploratory moves. If the step-size parameter is appropriately reduced over time, then the state values would converge to a set of probabilities. What are the two sets of probabilities computed when we do, and when we do not, learn from exploratory moves? Assuming that we do continue to make exploratory moves, which set of probabilities might be better to learn? Which would result in more wins?

	- Probability set with no learning from exploration, is the value of each state given optimal action taken from then on. This is assuming the step-size parameter is appropriately reduced and exploration is fixed
	- However, it is the expected value of each state including the active exploration policy when learning from exploration
	- Because the former reduces variance from sub-optimal future states, it is better to learn (resulting in more wins everything else being equal)
	- Example: Win one game of chess in one move, but if you make another move your opponent wins, that doesn't make it a bad state

1.5 Can you think of other ways to improve the reinforcement learning player? Can you think of any better way to solve the tic-tac-toe problem as posed?

	- Speed can be improved by decaying the old updates if the player is adapting over time
	- Depending on the variance in the opponent's actions, you can change the exploration rate/learning
