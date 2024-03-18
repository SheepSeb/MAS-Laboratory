from environment import *
import time


class MyAgent(BlocksWorldAgent):

    def __init__(self, name: str, desired_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)
        self.desired_state = desired_state
        self.current_plan: List[BlocksWorldAction] = []
        self.beliefs: BlocksWorldPerception = None
        self.completed_stacks = 0

    def response(self, perception: BlocksWorldPerception):
        # TODO: revise beliefs; if necessary, make a plan; return an action.
        if perception.current_world == self.desired_state:
            return NoAction()
        else:
            if self.revise_beliefs(perception.current_world) is True:
                self.beliefs = perception.current_world
                print("Current plan: ", self.current_plan.pop(0))
                return self.current_plan.pop(0)
            else:
                self.beliefs = perception
                print("Arm empty: ", ArmEmpty())
                if ArmEmpty():
                    self.current_plan = self.plan()
                return self.current_plan.pop(0)
        # raise NotImplementedError("not implemented yet; todo by student")
        # return NoAction()

    def revise_beliefs(self, perceived_world_state: BlocksWorld):
        # TODO: check if what the agent knows corresponds to what the agent sees
        # raise NotImplementedError("not implemented yet; todo by student")
        if self.beliefs is None:
            return False
        print("Perceived world state:\n", perceived_world_state)
        print("Beliefs:\n", self.beliefs.current_world)
        print("If perceived world state is equal to beliefs: ", perceived_world_state == self.beliefs.current_world)
        if perceived_world_state == self.beliefs.current_world:
            return True
        return False

    def plan(self) -> List[BlocksWorldAction]:
        # TODO: return a new plan, as a sequence of `BlocksWorldAction' instances, based on the agent's knowledge.
        # print("Agent %s is planning." % str(self))
        action_list: List[BlocksWorldAction] = []
        current_world = self.beliefs.current_world
        desired_world = self.desired_state

        current_stacks = current_world.get_stacks()
        desired_stacks = desired_world.get_stacks()

        print("Current stacks: ", current_stacks)
        print("Desired stacks: ", desired_stacks)

        # Remove the stacks that are already in the desired state
        for stack in current_stacks:
            if stack in desired_stacks:
                blocks = stack.get_blocks()
                for block in blocks:
                    action_list.append(Lock(block))
                desired_stacks.remove(stack)
            else:
                action_list.append(Unstack(stack.get_top_block(), stack.get_below(stack.get_top_block())))
                action_list.append(PutDown(stack.get_top_block()))

        print("Desired stacks after removal: ", desired_stacks)

        print("Action list: ", action_list)

        if action_list:
            return action_list
        elif not ArmEmpty():
            print("Arm not empty")
        return [NoAction()]

    def status_string(self):
        # TODO: return information about the agent's current state and current plan.
        # return str(self) + " : PLAN MISSING"
        if self.plan():
            return str(self) + " : PLAN " + str(self.plan())
        return str(self) + " : PLAN MISSING"


class Tester(object):
    STEP_DELAY = 0.5
    TEST_SUITE = "tests/0e-large/"

    EXT = ".txt"
    SI = "si"
    SF = "sf"

    DYNAMICITY = .0

    AGENT_NAME = "*A"

    def __init__(self):
        self._environment = None
        self._agents = []

        self._initialize_environment(Tester.TEST_SUITE)
        self._initialize_agents(Tester.TEST_SUITE)

    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(BlocksWorld(input_stream=input_stream))

    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        agent_states = {}

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MyAgent(Tester.AGENT_NAME, desires)

            agent_states[agent] = desires
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            print("Agent %s desires:" % str(agent))
            print(str(desires))

    def make_steps(self):
        print("\n\n================================================= INITIAL STATE:")
        print(str(self._environment))
        print("\n\n=================================================")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            print(str(self._environment))

            for ag in self._agents:
                print(ag.status_string())

            nr_steps += 1

            print("\n\n================================================= STEP %i completed." % nr_steps)

        print("\n\n================================================= ALL STEPS COMPLETED")


if __name__ == "__main__":
    tester = Tester()
    tester.make_steps()
