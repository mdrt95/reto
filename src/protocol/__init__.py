"""External-protocol adapters that wrap the bounded core agent service.

Each module here re-shapes one third-party HTTP contract onto the same
``AgentService`` the first-party ``/api/chat`` route uses. Adapters never change
agent, tool, guardrail, or grounding behavior; they only translate the wire
format at the edge (decision D-001).
"""
