"""
A collection of classes and functions for playing certain types of
games. Specifically, an implementation of the MCTS algorithm.
"""
import copy
import numpy as np
import random, queue
import re
from math import sqrt, log
from random import sample
from syntaxeval import Analyzer
from sklearn.utils import shuffle
from random import uniform


def apply_temperature(distribution, temperature=1.0):
    logits = np.log(distribution)
    logits = logits * temperature
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def eval_actions(actions, analyzer, id_to_token):
    ss = []
    sents = ''.join([id_to_token(int(action)) for action in actions]).replace("▁","")
    for sent in re.split('[。?!]', sents):
        if not sent.strip():
            continue
        sent = sent + "。"

        try:
            result = analyzer(sent)
            if np.random.rand() < 0.3:
                print('[random simulation choice] {:s} (syntax: {:4f} semantics: {:4f})'.format(sent, result['probabilities']['syntax'], result['probabilities']['semantics']))
            ss.append(result['probabilities']['syntax'])
            #ss.append(result['probabilities']['semantics'])
        except:
            ss.append(0.0)

    return sum(ss)/len(ss)


def select_actions(normal_actions, max_selection):
    #some algorithms here
    dummy = normal_actions[:max_selection]
    return dummy

class Game(object):
    """
    Base class for multi-player adversarial games.
    """
    def actions(self, state):
        raise Exception('Method must be overridden.')

    def result(self, state, action):
        raise Exception('Method must be overridden.')

    def terminal(self, state):
        raise Exception('Method must be overridden.')

    def outcome(self, state):
        raise Exception('Method must be overridden.')


class NLGGame(Game):

    def __init__(self,
                 contexts=[],
                 start_depth=0,
                 max_depth=3,
                 max_selection=20,
                 weights=None,
                 policy=None,
                 id_to_token=None):
        self.max_depth = max_depth
        self.start_depth = start_depth
        self.contexts = contexts
        self.max_selection = max_selection
        self.weights = weights
        self.policy = policy
        self.id_to_token = id_to_token

    def actions(self, state):
        return select_actions(state.normal_actions, self.max_selection)

    def result(self, state, action):
        current_depth = state.current_depth + 1
        new_actions = copy.copy(state.actions)
        new_actions.append(action)
        new_state = State(
            current_depth=current_depth,
            actions=new_actions,
            analyzer=state.analyzer)
        return new_state

    def terminal(self, state):
        return state.current_depth >= self.max_depth

    def outcome(self, state):
        reward = eval_actions(state.actions,
                              state.analyzer.analyze,
                              self.id_to_token)
        return reward


class Action:
    def __init__(self, index, token):
        self.index = index
        self.token = token

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.index == other.index

    def __hash__(self):
        return hash(self.index)

    def __str__(self):
        return "token:{}, index:{}".format(self.token, self.index)


class State:
    def __init__(
            self,
            current_depth=0,
            actions=None,
            analyzer=None
    ):
        self.current_depth = current_depth
        self.actions = actions
        if analyzer:
            self.analyzer = analyzer
        else:
            self.analyzer = Analyzer()


class Node(object):
    def __init__(self, parent, action, state, game=None, P=0.0):
        if parent is None and game is None:
            raise Exception('No game provided')
        # Game
        self.game = game or parent.game
        # Structure
        self.parent    = parent
        self.children = None
        # Tree data
        self.action    = action
        self.state     = state
        # Search meta data
        self.P = P
        self.visits    = 0
        self.value     = 0.0
        self.Q = 0.0
        # Hyper parameter
        self.C_puct = 5
        self.expansion_threshold = 10
        self.temperature = 0.67
    
    def __iter__(self):
        """
        A generator function. Does a pre-order traversal over the nodes
        in the tree without using recursion.
        """
        active = Queue.Queue()
        active.put(self)
        while active.qsize() > 0:
            next = active.get()
            for _, child in next.children.items():
                if child is not None:
                    active.put(child)
            yield next

    def __len__(self):
        """
        Returns the number of nodes in the tree. This requires a
        traversal, so it has O(n) running time.
        """
        n = 0
        for node in self.traverse():
            n += 1
        return n

    @property
    def weight(self):
        """
        The weight of the current node.
        """
        if self.visits == 0:
            return 0
        return self.value / float(self.visits)

    def search_weight(self, c):
        """
        Compute the PUCT(UCT + Policy) search weight function for this node. Defined as:

            w = Q(v') / N(v') + c_puct * P * sqrt(N(v)) / 1 + N(v'))

        Where v' is the current node and v is the parent of the current node,
        and Q(x) is the total value of node x and N(x) is the number of visits
        to node x.
        """
        return self.Q + self.P * (sqrt(self.parent.visits)/(1+self.visits))

    def actions(self):
        """
        The valid actions for the current node state.
        """
        return self.game.actions(self.state)

    def result(self, action):
        """
        The state resulting from the given action taken on the current node
        state by the node player.
        """
        return self.game.result(self.state, action)

    def terminal(self):
        """
        Whether the current node state is terminal.
        """
        return self.game.terminal(self.state)

    def outcome(self):
        """
        Returns the game outcome for the given player (default is the node's
        player) for the node state.
        """
        return self.game.outcome(self.state)

    def fully_expanded(self):
        """
        Whether all child nodes have been expanded (instantiated). Essentially
        this just checks to see if any of its children are set to None.
        """
        return self.children is not None

    def expand(self):
        """
        Instantiates one of the unexpanded children (if there are any,
        otherwise raises an exception) and returns it.
        """
        if self.visits < self.expansion_threshold:
            return self

        p = self.game.policy(self.state.actions)
        top_n = np.argpartition(p, -10)[-10:]
        # renorm top_n probs
        top_np = apply_temperature(p[top_n], temperature=self.temperature)
        #top_np = np.log(p[top_n]) * self.temperature
        #top_np = top_np - top_np.max()
        #top_np = np.exp(top_np)
        #top_np = top_np/top_np.sum()
        child = int(np.where(top_n == p.argmax())[0])
        self.children = [Node(self, int(c[0]), self.result(int(c[0])), P=c[1]) for c in zip(top_n, top_np)]
        # self.children = [Node(self, e[0], self.result(e[0]), P=e[1]) for e in enumerate(p)]
        # child = np.random.choice(len(p), 1, p=p)[0]
        return self.children[child]

    def best_child(self, c=1/sqrt(2)):
        if not self.fully_expanded():
            raise Exception('Node is not fully expanded')

        return max(self.children, key=lambda x: x.search_weight(c))

    def best_action(self, c=1/sqrt(2)):
        """
        Returns the action needed to reach the best child from the current
        node.
        """
        return self.best_child(c).action

    def best_sequence(self):
        sequence = []
        children = self.children
        while children is not None:
            max_child = max(children, key=lambda x: x.visits)
            sequence.append(max_child)
            children = max_child.children
        return sequence

    def max_child(self):
        """
        Returns the child with the highest value.
        """
        return max(self.children.values(), key=lambda x: x.weight)

    def simulation(self):
        """
        Simulates the game to completion, choosing moves in a uniformly random
        manner. The outcome of the simulation is returns as the state value for
        the given player.
        """
        st = self.state
        while not self.game.terminal(st):
            p = self.game.policy(self.state.actions)
            p = apply_temperature(p, temperature=self.temperature)
            action = np.random.choice(len(p), 1, p=p)[0]
            st = self.game.result(st, action)
        return self.game.outcome(st)

    def dot_string(self, value=False, prettify=lambda x: x):
        """
        Returns the tree rooted at the current node as a string
        in dot format. Each node is labeled with its state, which
        is first run through prettify. If value is True, then
        the value is included in the node label.
        """
        output = ''
        output += 'digraph {\n'
        for node in self:
            # Define the node
            name = prettify(node.state)
            if value:
                name += '%s\\n' % node.value
            output += '\t"%s" [style="filled"]\n' % (name)
            # No edge into the root node
            if node.parent is None:
                continue
            # Add edge from node parent to node
            pname = prettify(node.parent.state)
            if value:
                pname += '%s\\n' % node.parent.value
            output += '\t"%s" -> "%s"\n' % (pname, name)
        output += '}'
        return output


def mcts_uct(game, state, n, report_every=10):
    """
    Implementation of the UCT variant of the MCTS algorithm.
    """
    root = Node(None, None, state, game)
    for step in range(n):
        # Tree Policy
        child = root
        while not child.terminal():
            if not child.fully_expanded():
                child = child.expand()
                break
            else:
                child = child.best_child()
        # Default Policy
        delta = child.simulation()
        # Backup
        while not child is None:
            child.visits += 1
            child.value += delta
            child.Q = child.value/child.visits
            child = child.parent

        if step % report_every == 0:
            bs = root.best_sequence()
            if not bs:
                continue
            print('Best Sequence (simulations={:d}):'.format(step))
            print('  Text: {:s}'.format(' '.join(game.id_to_token(token_id) for token_id in bs[-1].state.actions)))
            print('  IDs: ' + str(bs[-1].state.actions))
            print('  Policy: {:s}'.format('->'.join([str(node.P)[:5] for node in bs])))
            print('  Visits: {:s}'.format('->'.join([str(node.visits) for node in bs])))
            print('  Action Value: {:s}'.format('->'.join([str(node.value/node.visits)[:5] for node in bs])))

    return root.best_sequence()


def full_tree(game, state):
    """
    Creates a full game tree in which player moves first. The traversal is done
    in breadth-first order. The return value is the root node.
    """
    active = Queue.Queue()
    root = Node(None, None, state)
    active.put(root)

    current = None
    while active.qsize() > 0:
        current = active.get()
        # Assign value if this is a terminal node
        if game.terminal(current.state):
            continue
        # Explore children otherwise
        for action in game.actions(current.state):
            nstate = game.result(current.state, action)
            node = Node(current, action, nstate)
            current.children[action] = node
            active.put(node)
    return root


def minimax(game, state):
    """
    Applies the Minimax algorithm to the given game. Returns the
    root node with values assigned to each node in the game tree.
    """
    active = []
    
    root = full_tree(game, state)
    for node in root:
        active.append(node)
    
    current = None
    while active:
        current = active.pop()
        # Leaf (terminal) node
        if game.terminal(current.state):
            current.value = game.outcome(current.state)
            continue
        # Interior or root node
        values = tuple([i.value for i in current.children.values()])
        current.value = max(values)
    
    return root


def mcts(game, state, n):
    """
    Implementation of the UCT variant of the Monte Carlo Tree Search algorithm.
    """
    root = Node(None, None, state)
    unexplored = Queue.Queue()
    unexplored.put(root)

    for _ in xrange(n):
        # Quit early if we are out of nodes
        if unexplored.qsize() == 0:
            break
        # Add the new node to the tree
        current = unexplored.get()
        if current.parent is not None:
            current.parent.children[current.action] = current
        # Add the newly discovered nodes to the queue
        for action in game.actions(current.state):
            nstate = game.result(current.state, action)
            node = Node(current, action, nstate)
            unexplored.put(node)
        # Simulate the rest of the game from the current node
        cstate = current.state
        while not game.terminal(cstate):
            caction = random.choice(game.actions(cstate))
            cstate = game.result(cstate, caction)
        simvalue = game.outcome(cstate)
        # Back simulation value up to the root
        backup = current
        while backup is not None:
            backup.value += simvalue
            backup.visits += 1
            backup = backup.parent

    return root
