# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"


        # initialize score
        score = 0

        # reciporcal for distance to closest food (add)
        newFoodList = newFood.asList()
        closestFoodDistance = float("inf")
        for each in newFoodList:
            distance = manhattanDistance(newPos, each)
            if distance < closestFoodDistance:
                closestFoodDistance = distance
    
        score += 1/closestFoodDistance

        # add reciporcal of game score
        if successorGameState.getScore() != 0:
            score += 1/ successorGameState.getScore()

        # reciporcal for distance to closest ghost (subtract)
        # stract 10000 if ghost is very close
        ghostPositions = successorGameState.getGhostPositions()
        for i in range(len(ghostPositions)):
            ghostDistance = manhattanDistance(ghostPositions[i], newPos)
            if ghostDistance <= 1:
                score -= 10000
            else:
                if ghostDistance != 0:
                    score -= 1 / ghostDistance

        # remaining capsules,  add 100 if capsule is close
        remainingCapsules = successorGameState.getCapsules()
        for i in range(len(remainingCapsules)):
            capsuleDistance = manhattanDistance(remainingCapsules[i], newPos)
            if capsuleDistance <= 0:
                score += 100

        # subtract remaining food
        score -= len(newFoodList)
        
        # if going to new food will make all of them gone, add 100 to score
        if len(newFoodList) == 0:
            score += 100


        # return score
        return score 


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth) # this is the max depth we will reach (not start depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    # helper function maxValue() takes in game state, and also depth and agent
    # depth helps tell us if it is a terminal state
    # agent tells us which agent we are on (will always be pacman here)
    def maxValue(self, gameState, depth, agent):
 
        # check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # initialize value as neg infinity
        v = float("-inf")

        # check each action for best value
        actions = gameState.getLegalActions(agent)
        for each in actions:
            succState = gameState.generateSuccessor(agent, each)
            # go to agent 1 (first ghost to consider), stay on same depth level for ghost turn
            v = max(v, self.minValue(succState, depth, 1))
        return v


    # helper function minValue() takes in game state, and also depth and agent
    # depth helps tell us if it is a terminal state
    # agent tells us which agent we are on, and if we should call back to pacman, or another ghost
    def minValue(self, gameState, depth, agent):
        
        # check if we are at a terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # because we are dealing with multiple ghosts, add base case to
        # determine if we have more ghosts to check or if we will be going back to pacman
        returntoPacman = False
        if agent == gameState.getNumAgents() - 1:
            returntoPacman = True

        # initialize value as infinity
        v = float("inf")

        # check eavh action for best value
        actions = gameState.getLegalActions(agent)
        for each in actions:
            succState = gameState.generateSuccessor(agent, each)
            if not returntoPacman:
                # we have at least 1 more ghost to consider, depth level stays same
                v = min(v, self.minValue(succState, depth, agent + 1))
            else: 
                # proceed with regular call for maxValue(), back to pacman, open another depth level
                v = min(v, self.maxValue(succState, depth + 1, 0)) 
                
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # the minimax, maxValue, and minValue functions are based on the lecture notes
        # CS5100 Foundations in Artificial Intelligence, Fall 2020, Robert Platt

        # minimax needs to be defined inside getActions() since we can't forward declare functions
        # gets the argmax of the actions at the top level, starts recursion
        def minimax(gameState):

            # get all legal actions from root node
            actions = gameState.getLegalActions(0)

            # initialize variables for best value and action
            maxVal = float("-inf")
            action = None

            # "argmax" of possible actions' values
            for each in actions:
                succState = gameState.generateSuccessor(0, each)
                # agaent is 1 because moving to ghost turn, depth  = 0 because we are at start
                v = self.minValue(succState, 0, 1)
                # check if we found a better action, value
                if v > maxVal:
                    maxVal = v
                    action = each

            # return action that gets max value  
            return action

        # kick off minimax function
        return minimax(gameState)
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    # helper function maxValue() takes in game state, depth, agent, alpha, and beta
    # depth helps tell us if it is a terminal state
    # agent tells us which agent we are on (will always be pacman here)
    def maxValue(self, gameState, depth, agent, alpha, beta):
 
        # check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # initialize value as neg infinity
        v = float("-inf")

        # check each action for best value
        actions = gameState.getLegalActions(agent)
        for each in actions:
            succState = gameState.generateSuccessor(agent, each)
            # go to agent 1 (first ghost to consider), stay on same depth level for ghost turn
            v = max(v, self.minValue(succState, depth, 1, alpha, beta))
            # check for pruning
            if v > beta: # lecture code uses >= but this only works with >
                return v
            alpha = max(alpha, v)
        return v


    # helper function minValue() takes in game state, depth, agent, alpha, beta
    # depth helps tell us if it is a terminal state
    # agent tells us which agent we are on, and if we should call back to pacman, or another ghost
    def minValue(self, gameState, depth, agent, alpha, beta):
        
        # check if we are at a terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # because we are dealing with multiple ghosts, add base case to
        # determine if we have more ghosts to check or if we will be going back to pacman
        returntoPacman = False
        if agent == gameState.getNumAgents() - 1:
            returntoPacman = True

        # initialize value as infinity
        v = float("inf")

        # check each action for best value
        actions = gameState.getLegalActions(agent)
        for each in actions:
            succState = gameState.generateSuccessor(agent, each)
            if not returntoPacman:
                # we have at least 1 more ghost to consider, depth level stays same
                v = min(v, self.minValue(succState, depth, agent + 1, alpha, beta))
            else: 
                # proceed with regular call for maxValue(), back to pacman, open another depth level
                v = min(v, self.maxValue(succState, depth + 1, 0, alpha, beta)) 
            
            # check for pruning
            if v < alpha: # lecture code uses <= but only < works here
                return v
            beta = min(beta, v)
                
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # alpha beta search helper function, derived from lecture pseudocode
        def alphaBetaSearch(gameState):

            # get actions to check from
            actions = gameState.getLegalActions(0)

            # initialize alpha and beta values
            alpha = float("-inf")
            beta = float("inf")

            action = None

            # "argmax" of possible actions' values
            for each in actions:
                succState = gameState.generateSuccessor(0, each)
                # agent is 1 because moving to ghost turn, depth  = 0 because we are at start
                v = self.minValue(succState, 0, 1, alpha, beta)
                # we check against alpha here (unlike minimax which checks the max val from each action)
                if v > alpha:
                    alpha = v
                    action = each
 
            return action
        
        # kick off alphaBetaSearch function
        return alphaBetaSearch(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # helper function maxValue() takes in game state, and also depth and agent
    # depth helps tell us if it is a terminal state
    # agent tells us which agent we are on (will always be pacman here)
    def maxValue(self, gameState, depth, agent):
 
        # check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # initialize value as neg infinity
        v = float("-inf")

        # check each action for best value
        actions = gameState.getLegalActions(agent)
        for each in actions:
            succState = gameState.generateSuccessor(agent, each)
            # go to agent 1 (first ghost to consider), stay on same depth level for ghost turn
            v = max(v, self.minValue(succState, depth, 1))
        return v


    # helper function minValue() takes in game state, and also depth and agent
    # depth helps tell us if it is a terminal state
    # agent tells us which agent we are on, and if we should call back to pacman, or another ghost
    # since ghosts are acting randomly, this method will now be considered as "chance" nodes, 
    # rather than "min" nodes
    def minValue(self, gameState, depth, agent):
        
        # check if we are at a terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # because we are dealing with multiple ghosts, add base case to
        # determine if we have more ghosts to check or if we will be going back to pacman
        returntoPacman = False
        if agent == gameState.getNumAgents() - 1:
            returntoPacman = True

        # initialize value as infinity
        v = float("inf")

        # check each action for best value
        actions = gameState.getLegalActions(agent)
        # initialize the number of chance nodes we have to use to find average in loop
        totalChanceNodes = float(len(actions))
        sumChanceNodes = 0
        for each in actions:
            succState = gameState.generateSuccessor(agent, each)
            if not returntoPacman:
                # we have at least 1 more ghost to consider, depth level stays same
                v = self.minValue(succState, depth, agent + 1)
                sumChanceNodes = sumChanceNodes + v
            else: 
                # proceed with regular call for maxValue(), back to pacman, open another depth level
                v = self.maxValue(succState, depth + 1, 0)
                sumChanceNodes = sumChanceNodes + v
                
        # calc average
        return sumChanceNodes / totalChanceNodes

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # expectimax derived from lectures
        def expectimax(gameState):
            # get all actions in list
            actions = gameState.getLegalActions(0)

            # initialize variables to find best value, action
            maxVal = float("-inf")
            action = None

            # "argmax" of possible actions' values
            for each in actions:
                succState = gameState.generateSuccessor(0, each)
                # agaent is 1 because moving to ghost turn, depth  = 0 because we are at start
                v = self.minValue(succState, 0, 1)
                if v > maxVal:
                    maxVal = v
                    action = each

            # return action that gets max value  
            return action


        # kick off expectimax function
        return expectimax(gameState)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I start off by initializing the score to 0. Next, I find the closest
    food to pacman and add the reciporcal of that distance to the score. I then add
    the reciporical of the current game score to the score. After this, I check the
    current ghost positions; if a ghost is less than or equal to 1 distance away, I
    subtract 10000 from the score. For any other distance, I subtract only 10. I then check the 
    remaining power capsules. If one is close, I add 100 to the score. Finally,
    I subtract the number of remaining foods from the score, and return the score. 
    """
    "*** YOUR CODE HERE ***"


    # initialize score
    score = 0

    # reciporcal for distance to closest food (add)
    newFoodList = currentGameState.getFood().asList()
    closestFoodDistance = float("inf")
    for each in newFoodList:
        distance = manhattanDistance(currentGameState.getPacmanPosition(), each)
        if distance < closestFoodDistance:
            closestFoodDistance = distance
    
    score += 1/closestFoodDistance

    # add reciporcal of game score
    if currentGameState.getScore() != 0:
        score += 1/ currentGameState.getScore()

    # reciporcal for distance to closest ghost (subtract)
    ghostPositions = currentGameState.getGhostPositions()
    for i in range(len(ghostPositions)):
        ghostDistance = manhattanDistance(ghostPositions[i], currentGameState.getPacmanPosition())
        if ghostDistance <= 1:
            score -= 10000
        else:
            if ghostDistance != 0:
                score -= 1 / ghostDistance

    # add distance of capsule if close by
    remainingCapsules = currentGameState.getCapsules()
    for i in range(len(remainingCapsules)):
        capsuleDistance = manhattanDistance(remainingCapsules[i], currentGameState.getPacmanPosition())
        if capsuleDistance <= 0:
            score += 100

    # subract number of remaining foods
    score -= len(newFoodList)

    # if len(newFoodList) == 0:
    #     score += 100

    # return score
    return score 

# Abbreviation
better = betterEvaluationFunction
