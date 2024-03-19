from environment import *
import time


def process_stack(stack):
    processed_blocks = list(dict.fromkeys(stack.get_blocks()))
    return processed_blocks


class MyAgent(BlocksWorldAgent):

    def __init__(self, name: str, desired_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)
        self.desired_state = desired_state
        self.current_plan: List[BlocksWorldAction] = []
        self.beliefs: BlocksWorld = None
        self.locked_blocks = []
        self.arm_empty = True
        self.flag_lock = False

    def response(self, perception: BlocksWorldPerception):
        # TODO: revise beliefs; if necessary, make a plan; return an action.
        print("Current plan: " + str(self.current_plan))
        self.revise_beliefs(perception.current_world)
        if self.current_plan is None or len(self.current_plan) == 0:
            self.current_plan = self.plan()
        print("After Current plan: " + str(self.current_plan))
        if len(self.current_plan) > 0:
            action = self.current_plan.pop(0)
            if isinstance(action, Lock):
                block = action.get_argument()
                self.desired_state.lock(block)
            elif isinstance(action, Unstack):
                self.arm_empty = False
            elif isinstance(action, PutDown):
                self.arm_empty = True
            elif isinstance(action, PickUp):
                self.arm_empty = False
            elif isinstance(action, Stack):
                self.arm_empty = True
                self.flag_lock = True
            return action
        else:
            return NoAction()
        pass

    def revise_beliefs(self, perceived_world_state: BlocksWorld):
        # TODO: check if what the agent knows corresponds to what the agent sees.
        # print("Beliefs: ",self.beliefs)
        # print("Perceived: ",perceived_world_state)
        # print("Beliefes vs percevied ",self.beliefs != perceived_world_state)
        if self.beliefs is None or self.beliefs != perceived_world_state:
            if self.beliefs is None:
                self.beliefs = perceived_world_state
            completed = 0
            for stack in self.desired_state.stacks:
                for s in perceived_world_state.stacks:
                    if stack.get_top_block() == s.get_top_block() and s.is_locked(stack.get_top_block()):
                        completed += 1
            print("Completed: ", completed)
            print("Len: ", len(self.desired_state.stacks))
            if completed == len(self.desired_state.stacks):
                self.current_plan = [AgentCompleted()]
                return
            self.beliefs = perceived_world_state
            if self.arm_empty and not self.flag_lock:
                self.current_plan = self.plan()
                self.flag_lock = False
        else:
                self.beliefs = perceived_world_state
        pass

    def plan(self) -> List[BlocksWorldAction]:

        # TODO: return a new plan, as a sequence of `BlocksWorldAction' instances, based on the agent's knowledge.
        action_plan = []
        # print("Desired state: " + str(self.desired_state.stacks))
        # print("Beliefs: " + str(self.beliefs.stacks))
        complete = 0
        if complete == len(self.desired_state.stacks):
            return [AgentCompleted()]

        for stack in self.desired_state.stacks:
            for current_stacks in self.beliefs.stacks:
                current_blocks = process_stack(current_stacks)
                if stack.get_blocks() == current_blocks:
                    # print("Desired stack found: " + str(stack))
                    # print("Current stack found: " + str(current_stacks))
                    for block in stack.get_blocks():
                        if not stack.is_locked(block):
                            action_plan.append(Lock(block))

        for stack in self.beliefs.stacks:
            top_block = stack.get_top_block()
            if not stack.is_locked(top_block) and stack.get_below(top_block) is not None:
                action_plan.append(Unstack(top_block, stack.get_below(top_block)))
                action_plan.append(PutDown(top_block))

        for stack in self.beliefs.stacks:
            top_block = stack.get_top_block()
            for desired_stack in self.desired_state.stacks:
                desired_bot_block = desired_stack.get_bottom_block()
                if top_block == desired_bot_block and not stack.is_locked(top_block):
                    action_plan.append(Lock(top_block))

        for desired_stack in self.desired_state.stacks:
            next_after_bot = desired_stack.get_above(desired_stack.get_bottom_block())
            for stack in self.beliefs.stacks:
                if stack.get_top_block() == next_after_bot and not stack.is_locked(next_after_bot):
                    action_plan.append(PickUp(next_after_bot))
                    action_plan.append(Stack(next_after_bot, desired_stack.get_below(next_after_bot)))
                    action_plan.append(Lock(next_after_bot))

        for desired_stack in self.desired_state.stacks:
            top_block = desired_stack.get_top_block()
            if not desired_stack.is_locked(top_block):
                action_plan.append(PickUp(top_block))
                action_plan.append(Stack(top_block, desired_stack.get_below(top_block)))
                action_plan.append(Lock(top_block))

        if len(action_plan) != 0:
            return action_plan
        return [NoAction()]
        pass

    def status_string(self):
        # TODO: return information about the agent's current state and current plan.
        # return str(self) + " : PLAN MISSING"
        if self.current_plan is None or len(self.current_plan) == 0:
            return str(self) + " : PLAN MISSING"
        else:
            return str(self) + " : PLAN AVAILABLE : " + str(self.current_plan)


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
