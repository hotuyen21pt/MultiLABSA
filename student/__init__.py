"""MERA-XQUAD student (MultiLABSA.docx §4.7, §4.9, §4.10).

Hybrid student = multilingual generative backbone (mT5) + auxiliary heads
(AT/OT span, AC/SP classification, AT–OT relation, implicit AT/OT NULL heads),
trained by a multi-objective loss over routed pseudo-labels, with an EMA teacher
and a language curriculum driving self-training rounds.
"""
