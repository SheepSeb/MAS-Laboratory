from environment import *
import time


class MyAgent(BlocksWorldAgent):

    def __init__(self, name: str, desired_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)
        self.desired_state = desired_state
        self.desired_state_stack = self.desired_state.get_stacks()
        self.current_plan: List[BlocksWorldAction] = []
        self.beliefs: BlocksWorldPerception = None
        self.completed_stacks = 0
        self.arm_empty = True

    def response(self, perception: BlocksWorldPerception):
        # TODO: revise beliefs; if necessary, make a plan; return an action.
        self.beliefs = perception
        # self.revise_beliefs(perception.current_world)
        # if len(self.plan()) == 0:
        #     return NoAction()
        # return self.plan().pop(0)
        print("Agent %s received perception: %s" % (str(self), str(perception.current_world.stacks)))
        print("Agent desired state: %s" % str(self.desired_state.stacks))

        return self.plan().pop(0)

    def revise_beliefs(self, perceived_world_state: BlocksWorld):
        # TODO: check if what the agent knows corresponds to what the agent sees.
        # If not, update the agent's beliefs.
        # Get the stacks and check
        perceived_stacks = perceived_world_state.get_stacks()
        # Remove the duplicates H H L P
        correct_stacks = []
        for stack in perceived_stacks:
            blc_stack = set(stack.get_blocks())
            correct_stacks.append(BlockStack(stack.get_bottom_block(), blc_stack))
        pass

    def __lock_correct_block(self, stacks: List[BlockStack]):
        for stack in stacks:
            if stack in self.desired_state_stack:
                for block in stack.get_blocks():
                    if stack.is_locked(block) is False:
                        return Lock(block)

        return NoAction()

    def plan(self) -> List[BlocksWorldAction]:
        # TODO: return a new plan, as a sequence of `BlocksWorldAction' instances, based on the agent's knowledge.

        action_list: List[BlocksWorldAction] = []
        current_world = self.beliefs.current_world

        self.current_plan = []
        if current_world == self.desired_state:
            return [AgentCompleted()]

        self.desired_state.lock(Block("B"))
        # Lock also in the desired state
        # Lock the correct block

        # Move the blocks to the correct position
        return [Lock(Block("B"))]

    def status_string(self):
        # TODO: return information about the agent's current state and current plan.
        # return str(self) + " : PLAN MISSING"
        if self.current_plan is not None and len(self.plan()) > 0:
            return str(self) + " : PLAN " + str(self.plan())
        return str(self) + " : PLAN MISSING"


class Tester(object):
    STEP_DELAY = 0.5
    TEST_SUITE = "tests/0e-large/"

    EXT = ".txt"
    SI = "si"
    SF = "sf"

    DYNAMICITY = .3

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
