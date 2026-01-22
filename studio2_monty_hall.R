# Problem 1: Monty Hall Simulation
# Summary: Write a function to simulate the Monty Hall problem
studio2_problem1 <- function(n_doors, n_trial, keep){
  
  win <- "Car" # the prize you want
  lose <- "Goat" # the prize you (probably) don't want
  results <- rep(0, n_trial)
  
  for (i in 1:n_trial) {
    # Set up doors: one car, rest goats
    doors <- c(rep(lose, n_doors - 1), win)
    doors <- sample(doors) # Randomize door positions
    
    # Player makes an initial choice
    player_choice <- sample(1:n_doors, 1)
    
    # Host opens a door with a goat, not the player's choice
    possible_doors_to_open <- which(doors == lose & seq_along(doors) != player_choice)
    host_opens <- sample(possible_doors_to_open, 1)
    
    final_choice <- player_choice
    if(!keep){
      # Player switches: can only choose from doors not originally picked and not opened by host
      final_choice_1 <- 1 - (doors[final_choice] == win)
      results[i] <- final_choice_1
    } else {
      results[i] <- doors[final_choice] == win
    }
  }
  
  return(mean(results))
}

studio2_problem1(n_doors=3, n_trial=1000, keep=TRUE)
studio2_problem1(n_doors=3, n_trial=1000, keep=FALSE)

library(ggplot2)

# Problem 2: Visualize your results
# Run your simulation for a number of times large enough (anything between 2000-10000)
# Plot your simulation results of keep=TRUE and keep=FALSE respectively
# ********* YOUR CODE BELOW HERE ***********
n_trial <- 5000

keep_prob <- studio2_problem1(3, n_trial, keep=TRUE)
switch_prob <- studio2_problem1(3, n_trial, keep=FALSE)

df <- data.frame(
  Strategy = c("Keep", "Switch"),
  Probability = c(keep_prob, switch_prob)
)

ggplot(df, aes(x=Strategy, y=Probability, fill=Strategy)) +
  geom_bar(stat="identity", width=0.5) +
  ylim(0, 1) +
  ggtitle("Monty Hall Win Probability: Keep vs Switch") +
  ylab("Estimated Win Probability")

# Problem 3: Visualize your results when you have more doors
# do a plot of estimated probability vs. number of doors from 3 - 100
# ********* YOUR CODE BELOW HERE ***********
door_range <- 3:100
n_trial <- 1000

switch_probs <- sapply(door_range, function(d) studio2_problem1(d, n_trial, keep=FALSE))

df <- data.frame(
  Doors = door_range,
  Probability = switch_probs,
  Strategy = "Switch"
)

ggplot(df, aes(x=Doors, y=Probability)) +
  geom_line(linewidth=1) +
  ggtitle("Monty Hall Win Probability vs Number of Doors") +
  ylab("Estimated Win Probability") +
  theme_minimal()
