import graphviz


# DFA that accepts strings that start with 'b' but don't contain 'bab'
def dfa_2a(format):
  # Create a Digraph object
  dfa = graphviz.Digraph(format=format)
  dfa.attr(rankdir='LR')
  # Define DFA states and transitions
  dfa.attr('node', shape='circle')
  dfa.node('q0', 'q0')
  dfa.node('q1', 'q1', peripheries='2')  # Accepting state
  dfa.node('q2', 'q2')

  # Initial state
  dfa.attr('node', shape='none')
  dfa.node('qi', '')

  # Transitions
  dfa.edge('qi', 'q0', label='')
  dfa.edge('q0', 'q1', label='b')
  dfa.edge('q1', 'q1', label='b')
  dfa.edge('q1', 'q2', label='a')
  dfa.edge('q2', 'q1', label='a')
  dfa.edge('q2', 'q2', label='b')

  # Return the Digraph object
  return dfa


def dfa_2b(format):
  # Create a Digraph object
  dfa = graphviz.Digraph(format=format)
  dfa.attr(rankdir='LR')

  # Define DFA states and transitions
  dfa.attr('node', shape='circle')
  dfa.node('q0', 'q0', peripheries='2')  # Accepting state
  dfa.node('q1', 'q1')
  dfa.node('q2', 'q2')

  # Initial states
  dfa.attr('node', shape='none')
  dfa.node('qi', '')

  # transitions for q0
  dfa.edge('qi', 'q0', label='')
  dfa.edge('q0', 'q1', label='a/b/c')
  dfa.edge('q1', 'q2', label='a/b/c')
  dfa.edge('q2', 'q0', label='a/b/c')

  return dfa


def dfa_2c(format):
  # Create a Digraph object
  dfa = graphviz.Digraph(format=format)
  dfa.attr(rankdir='LR')

  # Define DFA states and transitions
  dfa.attr('node', shape='circle')
  dfa.node('q0', 'q0')
  dfa.node('q1', 'q1')
  dfa.node('q2', 'q2')
  dfa.node('q3', 'q3', peripheries='2')  # Accepting state

  # Initial states
  dfa.attr('node', shape='none')
  dfa.node('qi', '')

  # transitions for q0
  dfa.edge('qi', 'q0', label='')
  dfa.edge('q0', 'q0', label='b')
  dfa.edge('q0', 'q1', label='a')
  dfa.edge('q1', 'q3', label='a')
  dfa.edge('q1', 'q2', label='b')
  dfa.edge('q2', 'q3', label='a')
  dfa.edge('q3', 'q3', label='a/b')

  return dfa


def dfa_3(format):
  # Define DFA M1 where n is a multiple of 3
  m1 = graphviz.Digraph('M1', format=format)
  m1.attr(rankdir='LR', size='8,5')
  m1.attr('node', shape='circle')

  # States for M1
  m1_states = ['q0', 'q1', 'q2']
  m1_start_state = 'q0'
  m1_accept_states = ['q0']
  m1_edges = [
    ('q0', 'a', 'q1'),
    ('q1', 'a', 'q2'),
    ('q2', 'a', 'q0'),
    ('q0', 'b', 'q0'),
    ('q1', 'b', 'q1'),
    ('q2', 'b', 'q2'),
  ]

  # Define the states
  for state in m1_states:
    if state in m1_accept_states:
      m1.node(state, peripheries='2')
    else:
      m1.node(state)

  # Add transitions
  for start, input, end in m1_edges:
    m1.edge(start, end, label=input)

  # Define DFA M2 where m is a multiple of 3
  m2 = graphviz.Digraph('M2', format=format)
  m2.attr(rankdir='LR', size='8,5')
  m2.attr('node', shape='circle')

  # States for M2
  m2_states = ['p0', 'p1', 'p2']
  m2_start_state = 'p0'
  m2_accept_states = ['p0']
  m2_edges = [
    ('p0', 'b', 'p1'),
    ('p1', 'b', 'p2'),
    ('p2', 'b', 'p0'),
    ('p0', 'a', 'p0'),
    ('p1', 'a', 'p1'),
    ('p2', 'a', 'p2'),
  ]

  # Define the states for M2
  for state in m2_states:
    if state in m2_accept_states:
      m2.node(state, peripheries='2')
    else:
      m2.node(state)

  # Add transitions for M2
  for start, input, end in m2_edges:
    m2.edge(start, end, label=input)

  m1.render('./a1/dfa_3a', view=False)
  m2.render('./a1/dfa_3b', view=False)

  m_union = graphviz.Digraph('M_union', format=format)
  m_union.attr(rankdir='LR', size='10,5')
  m_union.attr('node', shape='circle')

  # Product states combining M1 and M2 states
  union_states = [(q, p) for q in m1_states for p in m2_states]
  union_start_state = (m1_start_state, m2_start_state)
  union_accept_states = [(q, p) for q in m1_accept_states for p in m2_states] + [
    (q, p) for q in m1_states for p in m2_accept_states
  ]

  # Transitions for the union DFA
  union_edges = []
  for q1 in m1_states:
    for p1 in m2_states:
      for a in ['a', 'b']:
        # Find the next state for q1 in M1
        next_q1 = next((end for start, input, end in m1_edges if start == q1 and input == a), None)
        # Find the next state for p1 in M2
        next_p1 = next((end for start, input, end in m2_edges if start == p1 and input == a), None)
        # Add the transition to the union edges
        if next_q1 and next_p1:
          union_edges.append(((q1, p1), a, (next_q1, next_p1)))

  # Define the states for the union DFA
  for state in union_states:
    if state in union_accept_states:
      m_union.node(str(state), peripheries='2')
    else:
      m_union.node(str(state))

  # Add transitions for the union DFA
  for start, input, end in union_edges:
    m_union.edge(str(start), str(end), label=input)

  return m_union

def nfa(format):
  dfa = graphviz.Digraph('DFA from NFA', format=format)
  dfa.attr(rankdir='LR', size='8,5')

# Define the node attributes
  dfa.attr('node', shape='doublecircle')
  dfa.node('D3')
  dfa.node('D5')

  dfa.attr('node', shape='circle')
  dfa.node('start', shape='none')
  dfa.node('start', '')

# Define the DFA states based on the transition table given
  dfa_states = {
      'D0': '{q0}',
      'D1': '{q0, q1}',
      'D2': '{q0, q2}',
      'D3': '{q0, q1, q2}',
      'D4': '{q0, q3}',
      'D5': '{q0, q1, q2, q3}',
      'D6': '{q4}'
  }

# Define the edges/transitions according to the DFA transition table
  dfa.edge('start', 'D0', label='')
  dfa.edge('D0', 'D1', label='a')
  dfa.edge('D0', 'D0', label='b')
  dfa.edge('D1', 'D3', label='a')
  dfa.edge('D1', 'D2', label='b')
  dfa.edge('D2', 'D5', label='a')
  dfa.edge('D2', 'D4', label='b')
  dfa.edge('D3', 'D5', label='a')
  dfa.edge('D3', 'D4', label='b')
  dfa.edge('D4', 'D6', label='a,b')
  dfa.edge('D5', 'D6', label='a,b')
  dfa.edge('D6', 'D6', label='a,b')
  return dfa


m = {'a': dfa_2a, 'b': dfa_2b, 'c': dfa_2c, '3': dfa_3, '4': nfa}

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('question', type=str, help='Question number')
  parser.add_argument('--output-type', type=str, choices=['svg', 'png'], default='svg', help='Output file type')
  args = parser.parse_args()
  output_path = f'./a1/dfa_2{args.question}' if args.question in ['a', 'b', 'c', '3'] else f'./a1/dfa_4{args.question}'
  dfa = m[args.question](args.output_type)
  dfa.render(output_path, view=False)
