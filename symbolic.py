import matplotlib.pyplot as plt
import numpy as np
import heapq


class Planner:
    """
    Adapted from Robot Autonomy class
    """
    def __init__(self):
        self.Predicates = ['InHallway', 'InKitchen', 'InOffice', 'InLivingRoom', 'InGarden', 'OnRobot', 'Chopped']
        self.Objects = ['Robot', 'Fruit', 'Knife']

        self.ActionPre, self.ActionEff, self.ActionDesc = self.InitializePreconditionsAndEffects()

        self.InitialState, self.GoalState = self.InitializeStateAndGoal()

    def InitializePreconditionsAndEffects(self):
        nrObjects = len(self.Objects)
        nrPredicates = len(self.Predicates)

        ActionPre = []
        ActionEff = []
        ActionDesc = []

        ### Move to hallway
        for i in range(1, 5, 1):
            Precond = np.zeros([nrObjects, nrPredicates])
            Precond[0][0] = -1  # Robot not in hallway
            Precond[0][i] = 1  # Robot in i-th room

            Effect = np.zeros([nrObjects, nrPredicates])
            Effect[0][0] = 2.  # Robot in the hallway
            Effect[0][i] = -2.  # Robot not in the i-th room

            ActionPre.append(Precond)
            ActionEff.append(Effect)
            ActionDesc.append("Move to inHallway from " + self.Predicates[i])

        ### Move to room
        for i in range(1, 5, 1):
            Precond = np.zeros([nrObjects, nrPredicates])
            Precond[0][0] = 1  # Robot in the hallway
            Precond[0][i] = -1  # Robot not in the ith room

            Effect = np.zeros([nrObjects, nrPredicates])
            Effect[0][0] = -2.  # Robot not in the hallway
            Effect[0][i] = 2.  # Robot in the ith room

            ActionPre.append(Precond)
            ActionEff.append(Effect)
            ActionDesc.append("Move to " + self.Predicates[i] + " from InHallway")

        ###Cut fruit in kitchen
        Precond = np.zeros([nrObjects, nrPredicates])
        # NOTE: Robot in the kitchen, fruit in the kitchen, knife in the kitchen, fruit not chopped
        Precond[0][1] = 1  # Robot in kitchen
        Precond[1][1] = 1  # Fruit in kitchen
        Precond[2][1] = 1  # Knife in kitchen
        Precond[1][6] = -1  # Fruit not chopped

        Effect = np.zeros([nrObjects, nrPredicates])
        # NOTE: Fruit is chopped
        Effect[1][6] = 2  # Chop the fruit

        ActionPre.append(Precond)
        ActionEff.append(Effect)
        ActionDesc.append("Cut " + self.Objects[1] + " in InKitchen")

        ###Pickup object
        for i in range(0, 5, 1):
            for j in range(1, 3, 1):
                Precond = np.zeros([nrObjects, nrPredicates])
                Precond[0][i] = 1  # Robot in ith room
                Precond[j][i] = 1  # Object j in ith room
                Precond[j][5] = -1  # Object j not on robot

                Effect = np.zeros([nrObjects, nrPredicates])
                Effect[j][i] = -2  # Object j not in ith room
                Effect[j][5] = 2  # Object j on robot

                ActionPre.append(Precond)
                ActionEff.append(Effect)
                ActionDesc.append("Pick up " + self.Objects[j] + " from " + self.Predicates[i])

        ###Place object
        for i in range(0, 5, 1):
            for j in range(1, 3, 1):
                Precond = np.zeros([nrObjects, nrPredicates])
                Precond[0][i] = 1  # Robot in ith room
                Precond[j][i] = -1  # Object j not in ith room
                Precond[j][5] = 1  # Object j on robot

                Effect = np.zeros([nrObjects, nrPredicates])
                Effect[j][i] = 2.  # Object j in ith room
                Effect[j][5] = -2  # Object j not on robot

                ActionPre.append(Precond)
                ActionEff.append(Effect)
                ActionDesc.append("Place " + self.Objects[j] + " at " + self.Predicates[i])

        return ActionPre, ActionEff, ActionDesc

    def InitializeStateAndGoal(self):
        nrObjects = len(self.Objects)
        nrPredicates = len(self.Predicates)

        InitialState = -1 * np.ones([nrObjects, nrPredicates]) # Wait for clicks to place objects

        GoalState = np.zeros([nrObjects, nrPredicates])
        GoalState[0][1] = 1  # Robot is in the kitchen
        GoalState[1][1] = 1  # Fruit is in the kitchen
        GoalState[1][6] = 1  # Fruit is chopped

        return InitialState, GoalState

    def CheckCondition(self, state, condition):
        """
        Check if the condition is satisfied in the state
        """
        if (np.sum(np.multiply(state, condition)) - np.sum(np.multiply(condition, condition))) == 0:
            return True
        else:
            return False

    def CheckVisited(self, state, vertices):
        """
        Check if the state is already visited
        """
        for i in range(len(vertices)):
            if np.linalg.norm(np.subtract(state, vertices[i])) == 0:
                return True
        return False

    def ComputeNextState(self, state, effect):
        """
        Compute the next state by applying the effect
        """
        newstate = np.add(state, effect)
        return newstate

    def plan(self):
        vertices = []  # List of Visited States
        parent = []
        action = []
        cost2come = []
        pq = []  # Use heapq to implement priority queue

        heapq.heappush(pq, (0, 0))  # (cost, vertex_id)
        vertices.append(self.InitialState)
        parent.append(0)
        action.append(-1)
        cost2come.append(0)

        FoundPath = False
        while len(pq) > 0:
            # Get the element with the minimum cost
            heap_state = heapq.heappop(pq)  # Will pop the one with the smallest cost
            state_idx = heap_state[1]
            cost_to_state = cost2come[state_idx]  # Cost to go up to that state
            state = vertices[state_idx]  # The actual state is saved in the vertices
            # Check for Goal, use CheckCondition
            if self.CheckCondition(state, self.GoalState):
                FoundPath = True
                break
            # For all actions
            for action_idx in range(len(self.ActionPre)):
                # Check if the action is applicable, UseCheckCondition
                action_precond = self.ActionPre[action_idx]
                action_eff = self.ActionEff[action_idx]
                if self.CheckCondition(state, action_precond):
                    # Get the next state, use ComputeNextState
                    next_state = self.ComputeNextState(state, action_eff)
                    # If the next state is not visited, add it to the queue. Check visited using CheckVisited
                    if not self.CheckVisited(next_state, vertices):
                        vertices.append(next_state)
                        parent.append(state_idx)
                        action.append(action_idx)
                        cost2come_next = cost_to_state + 1
                        cost2come.append(cost2come_next)
                        heapq.heappush(pq, (cost2come_next, len(vertices) - 1))

        print("Path Found: ", FoundPath)
        # Extract Plan
        Plan = []
        PlanStates = []
        if FoundPath:
            x = state_idx
            while not x == 0:
                Plan.insert(0, action[x])
                PlanStates.insert(0, vertices[x])
                x = parent[x]
            # Print Plan
            print("States Explored: ", len(vertices))
            print("Plan Length: ", len(Plan))
            print()
            print("Plan:")
            for i in range(len(Plan)):
                print(self.ActionDesc[Plan[i]])
            return PlanStates

class House:
    """
    This class implements the GUI that shows the house and takes clicks from the user start the planner
    """
    def __init__(self, planner: Planner):
        # Start figure and define its limits
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(0, 500)
        self.ax.set_axis_off()
        self.ax.set_aspect('equal')
        self.ax.set_title("Click to place robot") # Always start with robot placement
        # Add the rooms of the house
        # Structure is name: [bottom-left corner, width, height, color]
        self.rooms = {"LivingRoom": [(0,0), 200, 250, "red"],
                 "Garden": [(0, 250), 200, 250, "blue"],
                 "Hallway": [(200, 0), 100, 500, "green"],
                 "Office": [(300, 0), 200, 250, "brown"],
                 "Kitchen": [(300, 250), 200, 250, "purple"]}
        # This will save the rooms as graphic entities, which is easier to work with
        self.patches = dict()
        for room in self.rooms.keys():
            self.add_room(room, *self.rooms[room])

        # Annotations that will move to display the objects planner and movement
        self.planner = planner # The GUI needs the backend planner in order to work
        self.object_annotations = dict() # This will save the names of the objects (and their info) to show on screen
        for element in self.planner.Objects:
            self.object_annotations[element] = None

        # Create the event handler for the button click
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def add_room(self, room_name, xy, width, height, color):
        # Add the rectangle for the room
        self.patches[room_name] = plt.Rectangle(xy, width, height, color=color)
        self.ax.add_patch(self.patches[room_name])
        # Place label off-center (ad hoc values used here)
        rx, ry = xy
        cx = rx + width/2.0
        cy = ry + 4.0*height/5.0
        self.ax.annotate(room_name, (cx, cy), color='w', weight='bold', fontsize=6, ha='center', va='center')

    def write_object_annotations(self, state):
        """
        Given a state, show on-screen where each object is in the house
        :param state: np.ndarray
        """
        # Clear annotations from the screen
        for annotation_name, annotation_value in self.object_annotations.items():
            if annotation_value is not None:
                annotation_value.remove()

        # Robot is written first
        # Robot is always in a room
        robot_room_idx = np.where(state[self.planner.Objects.index("Robot")] == 1)[0]
        if len(robot_room_idx) > 0:
            robot_room_idx = robot_room_idx[0]
            room_name = self.planner.Predicates[robot_room_idx]
            robot_room_name = room_name[2:] # Remove the "In" prefix
            # We annotate slightly off center to avoid stuff clustering on top of one another
            self.object_annotations["Robot"] = self.ax.annotate('Robot', self.patches[robot_room_name].get_center() + [0, 20],
                                                                color='w', fontsize=10, ha='center', va='center')
        # Knife can be either in a room or in the robot
        knife_room_idx = np.where(state[self.planner.Objects.index("Knife")] == 1)[0]
        if len(knife_room_idx) > 0:
            knife_room_idx = knife_room_idx[0]
            room_name = self.planner.Predicates[knife_room_idx]
            if room_name == "OnRobot":
                knife_room_name = robot_room_name # Knife will always be placed after robot, so this won't fail
            else:
                knife_room_name = room_name[2:]  # Remove the "In" prefix
            self.object_annotations["Knife"] = self.ax.annotate('Knife',
                                                                self.patches[knife_room_name].get_center(),
                                                                color='w', fontsize=10, ha='center', va='center')
        # Fruit can be either in a room or in the robot
        fruit_room_idx = np.where(state[self.planner.Objects.index("Fruit")] == 1)[0] # Will work even if fruit is chopped, as long as chopped is last
        if len(fruit_room_idx) > 0:
            fruit_room_idx = fruit_room_idx[0]
            room_name = self.planner.Predicates[fruit_room_idx]
            if room_name == "OnRobot":
                fruit_room_name = robot_room_name  # Fruit will always be placed after robot
            else:
                fruit_room_name = room_name[2:]  # Remove the "In" prefix
            self.object_annotations["Fruit"] = self.ax.annotate('Fruit',
                                                                self.patches[fruit_room_name].get_center() - [0, 20],
                                                                color='w', fontsize=10, ha='center', va='center')

    def on_click(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            # Find the room to which the click belongs
            for room_name in self.patches.keys():
                [x0, y0], _, [x1, y1], _ = self.patches[room_name].get_corners()
                if x0 <= x < x1 and y0 <= y < y1:
                    # First we annotate the robot, then the knife, then the fruit
                    # A bit hardcoded, but eh it works...
                    if self.object_annotations["Robot"] is None:
                        # Update the state
                        robot_idx = self.planner.Objects.index("Robot")
                        room_idx = self.planner.Predicates.index("In" + room_name)
                        self.planner.InitialState[robot_idx][room_idx] = 1
                        # Update the title on the screen and add the new object
                        self.ax.set_title("Click to place knife")
                        self.write_object_annotations(self.planner.InitialState)
                    elif self.object_annotations["Knife"] is None:
                        # Update the state
                        knife_idx = self.planner.Objects.index("Knife")
                        room_idx = self.planner.Predicates.index("In" + room_name)
                        self.planner.InitialState[knife_idx][room_idx] = 1
                        # Update the title on the screen and add the new object
                        self.ax.set_title("Click to place fruit")
                        self.write_object_annotations(self.planner.InitialState)
                    elif self.object_annotations["Fruit"] is None:
                        # Update the state
                        fruit_idx = self.planner.Objects.index("Fruit")
                        room_idx = self.planner.Predicates.index("In" + room_name)
                        self.planner.InitialState[fruit_idx][room_idx] = 1
                        # Update the title on the screen and add the new object
                        self.ax.set_title("Click to start planner")
                        self.write_object_annotations(self.planner.InitialState)
                    # Once all objects are placed, next click starts the planner
                    else:
                        states = self.planner.plan() # Plan returns list of states from start to goal
                        for state in states:
                            self.write_object_annotations(state) # Doesn't show "picking up" unfortunately
                            self.fig.canvas.draw()
                            plt.pause(1) # Arbitrary pause to make the state visible

                    self.fig.canvas.draw()
def main():
    planner = Planner()
    house = House(planner)
    plt.show()

if __name__ == "__main__":
    main()