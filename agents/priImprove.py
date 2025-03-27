from captureAgents import CaptureAgent
import random, math, heapq
from game import Directions
import util

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    return [OffensiveAgent(firstIndex), DefensiveAgent(secondIndex)]

class OffensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # self.start_position = gameState.getAgentPosition(self.index)
        # self.positionHistory = []
        # self.recentBlockedEntries = []
        # self.failedEntryAttempts = {}
        # self.historyLimit = 3
        # self.blockMemoryLimit = 5
 
        CaptureAgent.registerInitialState(self, gameState)
        self.start_position = gameState.getAgentPosition(self.index)
        self.positionHistory = []
        self.actionHistory = []  # Track recent actions
        self.historyLimit = 5  # Number of recent actions to track

    def chooseAction(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        legal_actions = [a for a in legal_actions if a != Directions.STOP]

        my_pos = gameState.getAgentPosition(self.index)
        food_list = self.getFood(gameState).asList()
        capsule_list = self.getCapsules(gameState)
        ghost_positions = self.getVisibleGhosts(gameState)
        carrying_food = gameState.getAgentState(self.index).numCarrying

        is_powered_up = any(
            gameState.getAgentState(enemy).scaredTimer > 2
            for enemy in self.getOpponents(gameState)
            if gameState.getAgentState(enemy).getPosition() in ghost_positions
        )

        best_score = float('-inf')
        best_action = Directions.STOP

        for action in legal_actions:
            print("Checking for legal action: ", action)
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            score = self.evaluateState(
                legal_actions, new_pos, food_list, capsule_list, ghost_positions,
                carrying_food, is_powered_up, gameState, action
            )
            if score > best_score:
                best_score = score
                best_action = action

            print(f"Score: {score}")
        
        print(f"Best action: {best_action} and score : {score}")
        next_state = gameState.generateSuccessor(self.index, best_action)
        self.positionHistory.append(next_state.getAgentPosition(self.index))
        if len(self.positionHistory) > self.historyLimit:
            self.positionHistory.pop(0)

        return best_action

    def isCycling1(self):
        """
        Detects if the agent is oscillating between positions (e.g., South-North-South-North).
        Returns True if a cycle is detected, False otherwise.
        """
        print("Check for cycling")
        print(self.positionHistory)
        if len(self.positionHistory) < self.historyLimit:
            return False

        # Check if the last few positions are repeating
        unique_positions = set(self.positionHistory)
        if len(unique_positions) <= 2:  # Oscillating between 2 positions
            print("*****", True)
            print(unique_positions)
            return True

        # Check for longer cycles (e.g., A -> B -> C -> A -> B -> C)
        for i in range(len(self.positionHistory) - 2):
            if self.positionHistory[i] == self.positionHistory[i + 2]:
                return True

        return False
    def isCycling(self, current_action):
        """
        Detects if the agent is oscillating between positions (e.g., South-North-South-North).
        Returns True if a cycle is detected, False otherwise.
        """
        if len(self.positionHistory) < 2:  # Need at least 2 positions to detect a cycle
            return False

        # Simulate the next position based on the current action
        next_pos = self.getNextPosition(self.positionHistory[-1], current_action)

        # Check for cycles of length 2 (e.g., A -> B -> A)
        if len(self.positionHistory) >= 2:
            if next_pos == self.positionHistory[-2]:
                return True

        # Check for cycles of length 3 (e.g., A -> B -> C -> A)
        if len(self.positionHistory) >= 3:
            if next_pos == self.positionHistory[-3]:
                return True

        # Check for cycles of length 4 (e.g., A -> B -> C -> D -> A)
        if len(self.positionHistory) >= 4:
            if next_pos == self.positionHistory[-4]:
                return True

        # Check for cycles of length 5 (e.g., A -> B -> C -> D -> E -> A)
        if len(self.positionHistory) >= 5:
            if next_pos == self.positionHistory[-5]:
                return True

        return False
    def evaluateState(self, legal_actions, my_pos, food_list, capsule_list,
                      ghost_positions, carrying_food, is_powered_up, gameState, action):
        features = self.getFeatures(
            legal_actions, my_pos, food_list, capsule_list, ghost_positions,
            carrying_food, is_powered_up, gameState, action
        )
        weights = self.getWeights(my_pos, carrying_food, ghost_positions, is_powered_up)
        
        return features * weights

    def getFeatures(self, legal_actions, my_pos, food_list, capsule_list,
                    ghost_positions, carrying_food, is_powered_up, gameState, action):
        situation = self.classifySituation(my_pos, carrying_food, ghost_positions, is_powered_up, gameState)

        if situation == "danger":
            return self.getFeaturesDanger(my_pos, ghost_positions, legal_actions, capsule_list, gameState, action)
        elif situation == "return":
            return self.getFeaturesReturn(my_pos, ghost_positions, gameState, action)
        elif situation == "power":
            return self.getFeaturesPower(my_pos, ghost_positions, food_list, gameState)
        else:
            return self.getFeaturesForage(my_pos, food_list, ghost_positions, capsule_list, gameState, action, carrying_food)

    def getSafeDirections(self, my_pos, ghost_positions, gameState):
        """
        Returns directions from current position that are away from nearby ghosts.
        """
        directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        safe_dirs = []

        for action in directions:
            next_pos = self.getNextPosition(my_pos, action)
            if not gameState.hasWall(int(next_pos[0]), int(next_pos[1])):
                if all(self.getMazeDistance(next_pos, ghost) > 2 for ghost in ghost_positions):
                    safe_dirs.append(action)

        return safe_dirs


    def classifySituation(self, my_pos, carrying_food, ghost_positions, is_powered_up, gameState):
        
            # print("FORAGE")
            # return "forage"
        #if self.isAcrossBorder(my_pos):
        if ghost_positions:
            min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions])
            if min_ghost_dist <= 3 and carrying_food >= 2 and not is_powered_up:
                return "danger"
            # elif min_ghost_dist <= 5 and not is_powered_up:
            #     return "danger"
        #Increase threshold dynamically if no visible ghosts
        if not ghost_positions:
            if carrying_food >= 5:
                print("RETURN")
                return "return"
        if carrying_food >= 3:
            print("RETURN")
            return "return"
        if is_powered_up:
            print("POWER")
            return "power"
        print("FORAGE")
        return "forage"

    def getFeaturesDanger(self, my_pos, ghost_positions, legal_actions, capsule_list, gameState, action):
        features = util.Counter()
        min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions]) if ghost_positions else float('inf')
 
        features["closeGhost"] = 1.0 / (min_ghost_dist + 1)
        if min_ghost_dist <= 1:
            features["deathZone"] = 1.0

        if len(legal_actions) <= 2:
            features["escapePotential"] = self.evaluateEscapePotential(my_pos, ghost_positions, legal_actions)

        if capsule_list:
            min_capsule_dist = min([self.getMazeDistance(my_pos, c) for c in capsule_list])
            if min_capsule_dist <= 1:
                features["capsulePriority"] = 1.0
            elif min_capsule_dist <= 5:
                features["capsuleDistance"] = 1.0 / (min_capsule_dist + 1)

        border_positions = self.getFriendlyBorders()
        if border_positions:
            min_border_distance = min([self.getMazeDistance(my_pos, b) for b in border_positions])
            features["invBorderDistance"] = 1.0 / (min_border_distance + 1)
            for border in border_positions:
                for ghost in ghost_positions:
                    if self.getMazeDistance(ghost, border) <= 2:
                        features["borderBlocked"] = 1.0
                        break
        
        if self.isCycling(action):
            features["cyclePenalty"] = 1.0
            
        if my_pos in self.positionHistory:
            features["repeatPenalty"] = 1.0
       
        return features

    def getFeaturesReturn(self, my_pos, ghost_positions, gameState, action):
        features = util.Counter()
        border_positions = self.getFriendlyBorders()
        if border_positions:
            min_border_distance = min([self.getMazeDistance(my_pos, b) for b in border_positions])
            features["invBorderDistance"] = 1.0 / (min_border_distance + 1)

        if ghost_positions:
            min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions])
            features["ghostThreatWhileReturning"] = 1.0 / (min_ghost_dist + 1)
            if min_ghost_dist <=5:
                if self.isCycling(action):
                    features["cyclePenalty"] = 1.0
                    

        if self.isBorderGuarded(my_pos, ghost_positions, gameState):
            features["borderGuarded"] = 1.0

        
        return features

    def getFeaturesPower(self, my_pos, ghost_positions, food_list, gameState):
        features = util.Counter()
        if ghost_positions:
            min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions])
            features["chaseGhost"] = 1.0 / (min_ghost_dist + 1)
        if food_list:
            dist = self.aStarSearch(my_pos, food_list, gameState)
            features["invFoodDistance"] = 1.0 / (dist + 1) 
        return features
    def isBorderGuarded(self, my_pos, ghost_positions, gameState):
        """
        Returns True if ghosts are near the border positions, indicating danger.
        """
        borders = self.getFriendlyBorders()
        for border in borders:
            for ghost in ghost_positions:
                if self.getMazeDistance(ghost, border) <= 2 and self.getMazeDistance(my_pos, border) <= 4:
                    return True
        return False
    def isEnemyBorderGuarded(self, my_pos, ghost_positions, gameState):
        """
        Returns True if ghosts are near the *enemy's* border positions, 
        indicating danger when trying to enter enemy territory.
        """
        width = self.getCurrentObservation().data.layout.width
        height = self.getCurrentObservation().data.layout.height
        walls = self.getCurrentObservation().getWalls().asList()

        # Enemy border x coordinate
        border_x = width // 2 if self.red else (width // 2) - 1

        # These are the border positions you'd cross to invade
        enemy_borders = [(border_x, y) for y in range(height) if (border_x, y) not in walls]

        for border in enemy_borders:
            for ghost in ghost_positions:
                if self.getMazeDistance(ghost, border) <= 2 and self.getMazeDistance(my_pos, border) <= 4:
                    return True
        return False
    
    def getSafeExitBorders(self, my_pos, ghost_positions, gameState):
        """
        Returns a list of safe exit borders (no ghosts nearby and accessible).
        """
        border_positions = self.getFriendlyBorders()
        safe_borders = []

        for border in border_positions:
            # Check if the border is accessible (no walls in the path)
            if not self.isPathBlocked(my_pos, border, gameState):
                # Check if the border is safe (no ghosts nearby)
                is_safe = True
                for ghost in ghost_positions:
                    if self.getMazeDistance(border, ghost) <= 2:  # Ghost is too close to the border
                        is_safe = False
                        break
                if is_safe:
                    safe_borders.append(border)

        return safe_borders

    def isPathBlocked(self, start_pos, end_pos, gameState):
        """
        Returns True if the path from start_pos to end_pos is blocked by walls, False otherwise.
        """
        walls = gameState.getWalls().asList()
        x1, y1 = start_pos
        x2, y2 = end_pos

        # Check if there's a wall in the straight-line path (simplified check)
        if x1 == x2:  # Vertical path
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if (x1, y) in walls:
                    return True
        elif y1 == y2:  # Horizontal path
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if (x, y1) in walls:
                    return True
        else:  # Diagonal path (not applicable in grid-based games)
            return True

        return False
    
    def getFeaturesForage(self, my_pos, food_list, ghost_positions, capsule_list, gameState, action, carry_food):
        features = util.Counter()
        if food_list:
            dist = min([self.getMazeDistance(my_pos, food) for food in food_list])
            features["invFoodDistance"] = 1.0 / (dist + 1)

        if ghost_positions:
            
            min_ghost_dist = min([self.getMazeDistance(my_pos, g) for g in ghost_positions])
            
            if min_ghost_dist <=5:
                features["closeGhost"] = 1.0 / (min_ghost_dist + 1)
                 
            if self.isCycling(action):
                features["cyclePenalty"] = 1.0 /  (min_ghost_dist + 1)
        safe_borders = self.getSafeExitBorders(my_pos, ghost_positions, gameState)
        
        if safe_borders:
                min_border_dist = min([self.getMazeDistance(my_pos, b) for b in safe_borders])
                features["safeExitBorderDistance"] = 1.0 / (min_border_dist + 1)  # Higher score for closer safe borders
        
        if self.isBorderGuarded(my_pos, ghost_positions, gameState):
            features["borderGuarded"] = 1.0
       
        #safe_borders = self.getSafeExitBorders(my_pos, ghost_positions, gameState)
    
        # if not self.isAcrossBorder(my_pos):
        #     blocked_entries = [entry for entry in self.getFriendlyBorders()
        #                     if any(self.getMazeDistance(entry, g) <= 2 for g in ghost_positions)]
        #     for entry in blocked_entries:
        #         if self.getMazeDistance(my_pos, entry) <= 2:
        #             features["entryBlocked"] = 1.0
        #             if entry not in self.recentBlockedEntries:
        #                 self.recentBlockedEntries.append(entry)
        #             if len(self.recentBlockedEntries) > self.blockMemoryLimit:
        #                 self.recentBlockedEntries.pop(0)

        #             self.failedEntryAttempts[entry] = self.failedEntryAttempts.get(entry, 0) + 1
        #             if self.failedEntryAttempts[entry] >= 2:
        #                 features["entryLoopPenalty"] = 1.0

        #     alternate_entries = [e for e in self.getFriendlyBorders() if e not in blocked_entries]
        #     if alternate_entries:
        #         min_dist = min([self.getMazeDistance(my_pos, e) for e in alternate_entries])
        #         features["alternateEntry"] = 2.0 / (min_dist + 1)

        
        # if my_pos in self.positionHistory:
        #     features["repeatPenalty"] = 1.0

        return features

    def getWeights(self, my_pos, carrying_food, ghost_positions, is_powered_up):
        min_ghost_distance = min([self.getMazeDistance(my_pos, g) for g in ghost_positions]) if ghost_positions else float('inf')
        
        weights = util.Counter()

        if is_powered_up:
            return {"chaseGhost": 300, "invFoodDistance": 20}

        # if not ghost_positions:
        #     return {
        #         "invFoodDistance": 100,
        #         "capsuleDistance": 30,
        #         "repeatPenalty": -80,
        #         "alternateEntry": 100
        #     }

        # if min_ghost_distance <= 3:
        #     weights.update({
        #         "closeGhost": -300,
        #         "deathZone": -10000,
        #         "capsulePriority": 100,
        #         "capsuleDistance": 60,
        #         "escapePotential": 10,
        #         "invBorderDistance": 70,
        #         "borderBlocked": -150,
        #         "repeatPenalty": -100,
        #         "safeDirectionBonus": -5,
        #         "entryBlocked": -300,
        #         "entryLoopPenalty": -500,
        #         "alternateEntry": 100
        #     })
        
        elif carrying_food >= 3:
            weights.update({
                "invFoodDistance": 5,
                "closeGhost": -100,
                "invBorderDistance": 400,
                "ghostThreatWhileReturning": -150,
                # "repeatPenalty": -100,
                "borderGuarded": -100,
                "cyclePenalty": -100, 
                
                "safeExitBorderDistance":80
            })
        else:
            weights.update({
                "invFoodDistance": 90,
                "closeGhost": -80,
                "capsuleDistance": 40,
                "invBorderDistance": 15,
                #"borderBlocked": -150,
                #"entryBlocked": -300,
                "entryLoopPenalty": -500,
                "repeatPenalty": -10,
                #"safeDirectionBonus": 2,
               # "borderGuarded": -150,
                "openSpace": 30,
                "cyclePenalty": -100, 
                "awayFromGhost": 20,
                #"safeExitBorderDistance":80
                #"alternateEntry": 100
            })

        return weights
    def getSafeBorders(self, my_pos, ghost_positions, gameState):
        """
        Returns a list of safe border positions (borders where no ghosts are nearby).
        """
        border_positions = self.getFriendlyBorders()
        safe_borders = []

        for border in border_positions:
            is_safe = True
            for ghost in ghost_positions:
                if self.getMazeDistance(border, ghost) <= 3:  # Ghost is too close to the border
                    is_safe = False
                    break
            if is_safe:
                safe_borders.append(border)

        return safe_borders
    def aStarSearch(self, start_pos, food_list, gameState):
        walls = gameState.getWalls()
        frontier = [(0, start_pos)]
        visited = set()
        cost_so_far = {start_pos: 0}

        while frontier:
            cost, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.add(current)
            if current in food_list:
                return cost
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = current[0] + dx, current[1] + dy
                if not walls[nx][ny]:
                    next_pos = (nx, ny)
                    new_cost = cost + 1
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        heuristic = min([abs(nx - fx) + abs(ny - fy) for fx, fy in food_list]) if food_list else 0
                        heapq.heappush(frontier, (new_cost + heuristic, next_pos))
        return float('inf')

    def evaluateEscapePotential(self, my_pos, ghost_positions, legal_actions):
        """
        Evaluates how well Pacman can escape by maximizing distance from ghosts.
        """
        escape_score = 0
        try:
            for action in legal_actions:
                next_pos = self.getNextPosition(my_pos, action)

                # **Find the minimum distance to any ghost after taking this action**
                
                min_ghost_distance = min([self.getMazeDistance(next_pos, ghost) for ghost in ghost_positions])

                # **Assign higher scores to moves that increase distance from ghosts**
                escape_score += 100 / (min_ghost_distance + 1)  # Higher distance = better escape move
        except:
            pass
        return escape_score / len(legal_actions)  # Average out escape scores
    
    def findAlternateEntryValue(self, my_pos, ghost_positions, gameState):
        width = self.getCurrentObservation().data.layout.width
        height = self.getCurrentObservation().data.layout.height
        walls = self.getCurrentObservation().getWalls().asList()

        # Enemy border x
        border_x = width // 2 if self.red else (width // 2) - 1
        enemy_borders = [(border_x, y) for y in range(height) if (border_x, y) not in walls]

        blocked = [b for b in enemy_borders if any(self.getMazeDistance(b, g) <= 2 for g in ghost_positions)]
        safe_entries = [b for b in enemy_borders if b not in blocked]

        if safe_entries:
            min_dist = min([self.getMazeDistance(my_pos, b) for b in safe_entries])
            return 1.0 / (min_dist + 1)
        return 0.0

    def getNextPosition(self, pos, action):
        x, y = pos
        dx, dy = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST:  (1, 0),
            Directions.WEST:  (-1, 0)
        }.get(action, (0, 0))
        return (x + dx, y + dy)

    def getVisibleGhosts(self, gameState):
        ghosts = []
        for enemy in self.getOpponents(gameState):
            state = gameState.getAgentState(enemy)
            if not state.isPacman and state.getPosition():
                ghosts.append(state.getPosition())
        return ghosts

    def isAcrossBorder(self, pos):
        mid_x = self.getCurrentObservation().data.layout.width // 2
        return pos[0] >= mid_x if self.red else pos[0] < mid_x

    def getFriendlyBorders(self):
        width = self.getCurrentObservation().data.layout.width
        height = self.getCurrentObservation().data.layout.height
        border_x = (width // 2) - 1 if self.red else width // 2
        walls = self.getCurrentObservation().getWalls().asList()
        return [(border_x, y) for y in range(height) if (border_x, y) not in walls]

##########
# DEFENSIVE AGENT #
##########

class DefensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start_position = gameState.getAgentPosition(self.index)
        self.positionHistory = []
        self.historyLimit = 3
        self.lastSeenInvaders = {}


    def chooseAction(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        my_pos = gameState.getAgentPosition(self.index)
        visible_invaders = self.getEnemyPacman(gameState)

        # Update last seen positions
        if visible_invaders:
            self.lastSeenInvaders = {i: pos for i, pos in visible_invaders.items()}
        elif self.lastSeenInvaders:
            # Use last known invader positions if we lost sight
            visible_invaders = self.lastSeenInvaders

        border_positions = []
        if not visible_invaders:
            border_positions = self.getStrategicBorderPositions(gameState, visible_invaders)


        best_score = float('-inf')
        best_action = Directions.STOP

        for action in legal_actions:
            successor = gameState.generateSuccessor(self.index, action)
            new_pos = successor.getAgentPosition(self.index)
            features = self.getFeatures(new_pos, visible_invaders, border_positions, gameState)
            weights = self.getWeights()
            score = features * weights

            if score > best_score:
                best_score = score
                best_action = action

        next_state = gameState.generateSuccessor(self.index, best_action)
        self.positionHistory.append(next_state.getAgentPosition(self.index))
        if len(self.positionHistory) > self.historyLimit:
            self.positionHistory.pop(0)

        return best_action

    
    def getFeatures(self, my_pos, invader_dict, border_positions, gameState):
        features = util.Counter()

        features["onDefense"] = 1 if not self.isAcrossBorder(my_pos) else 0

        if invader_dict:
            positions = list(invader_dict.values())
            features["numInvaders"] = len(positions)
            min_invader_dist = min([self.getMazeDistance(my_pos, inv) for inv in positions])
            features["invaderDistance"] = min_invader_dist
            features['borderProximity'] = 0
        elif border_positions:
            min_border_dist = min([self.getMazeDistance(my_pos, b) for b in border_positions])
            features["borderProximity"] = min_border_dist

        # Defend capsules
        enemy_capsules = set(gameState.getCapsules()) - set(self.getCapsulesYouAreDefending(gameState))
        if enemy_capsules:
            min_capsule_dist = min([self.getMazeDistance(my_pos, c) for c in enemy_capsules])
            features["capsuleGuard"] = min_capsule_dist

        if my_pos in self.positionHistory:
            features["repeatPenalty"] = 1

        return features

    def getWeights(self):
        return util.Counter({
            "onDefense": 100,
            "numInvaders": -1000,
            "invaderDistance": -20,
            "borderProximity": -2,
            "capsuleGuard": -5,
            "repeatPenalty": -50
        })

    def getEnemyPacman(self, gameState):
        invaders = {}
        for i in self.getOpponents(gameState):
            state = gameState.getAgentState(i)
            if state.isPacman and state.getPosition() is not None:
                invaders[i] = state.getPosition()
        return invaders

    def isAcrossBorder(self, pos):
        width = self.getCurrentObservation().data.layout.width
        return pos[0] >= width // 2 if self.red else pos[0] < width // 2

    def getStrategicBorderPositions(self, gameState, invaders):
        width = self.getCurrentObservation().data.layout.width
        height = self.getCurrentObservation().data.layout.height
        walls = self.getCurrentObservation().getWalls().asList()
        border_x = (width // 2) - 1 if self.red else width // 2

        border_positions = [(border_x, y) for y in range(height) if (border_x, y) not in walls]

        if invaders:
            closest_invader = min(invaders, key=lambda inv: self.getMazeDistance(inv, self.start_position))
            closest_border = min(border_positions, key=lambda b: self.getMazeDistance(closest_invader, b))
            return [closest_border]

        return border_positions