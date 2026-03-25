# STAR Stories — Career Experience Bank

Use this file to store your interview stories in STAR format.
The RAG system will chunk and index these so you can query them semantically.

---

## Story: NAPA Interchange — Data Quality at Scale

**Tags:** data-driven, rollback, monitoring, scale, B2B
**Roles demonstrated:** Technical PM, cross-functional leadership, post-deployment monitoring

### Situation
NAPA had a 300M-row product interchange dataset mapping competitor part numbers
to NAPA equivalents. The existing system relied on legacy matching logic that
hadn't been validated at scale.

### Task
Lead the phased rollout of an updated interchange matching system, ensuring data
accuracy across the full catalog while minimizing disruption to B2B customers.

### Action
- Designed a phased rollout strategy to validate matches incrementally
- Built monitoring dashboards to track match quality post-deployment
- Identified ~2M rows of bad matches through anomaly detection in early phases
- Made the call to rollback the affected segment before it reached customers
- Partnered with engineering to add automated quality gates for future deploys

### Result
- Caught and rolled back bad matches before customer impact
- Established post-deployment monitoring as a standard practice
- Key lesson: monitoring after deployment is as critical as testing before it

---

## Story: NAPA Taxonomy — ML-Powered Categorization Cleanup

**Tags:** ML, taxonomy, classification, data-quality, legacy-systems
**Roles demonstrated:** Product vision, ML collaboration, stakeholder management

### Situation
NAPA's product taxonomy had 15-20 years of manual categorization decisions,
resulting in inconsistent categories, duplicates, and misclassified products
across hundreds of thousands of SKUs.

### Task
Define and lead a product initiative to clean up the taxonomy using ML-based
classification, replacing the manual process with a scalable, repeatable system.

### Action
- Partnered with data science to evaluate classification approaches (TF-IDF,
  rule-based, hybrid)
- Defined success metrics: accuracy threshold, coverage percentage, stakeholder
  sign-off process
- Managed stakeholder expectations across merchandising teams who had strong
  opinions about "their" categories
- Phased rollout: automated confident matches first, human-in-the-loop for
  edge cases

### Result
- Significant reduction in miscategorized products
- Established ML-based classification as the go-forward process
- Reduced manual categorization effort dramatically

---

## Story: Salesloft — AI/ML Feature Development

**Tags:** AI, ML, conversation-intelligence, B2B-SaaS, product-led
**Roles demonstrated:** AI PM, cross-functional execution, prioritization

### Situation
[Fill in your Salesloft story here]

### Task
[What were you asked to do?]

### Action
[What did you do? Be specific about your decisions and influence.]

### Result
[Measurable outcomes, lessons learned]

---

## Story: MeetingIQ — Building an AI-Powered Meeting Analyzer

**Tags:** agentic-AI, prompt-chaining, portfolio-project, Claude-API
**Roles demonstrated:** Technical building, AI architecture, product thinking

### Situation
Preparing for a Conversation Intelligence PM interview at CallRail, needed to
demonstrate hands-on understanding of how LLM pipelines work — not just talk
about them abstractly.

### Task
Build a working demo that shows multi-step LLM processing of meeting transcripts,
including summarization, sentiment analysis, coaching insights, and Q&A.

### Action
- Built MeetingIQ as a standalone HTML app using the Anthropic Messages API
- Designed a sequential prompt chain: summary → sentiment → coaching → Q&A
- Implemented real confidence scoring via self-assessment tags in each LLM call
- Added API error handling and loading states for production-quality UX
- Correctly identified architecture as prompt chaining (not true agentic behavior)
  and used that distinction as an interview talking point

### Result
- Working demo that processes meeting transcripts through 4 analysis stages
- Demonstrated technical depth beyond typical PM portfolio pieces
- Used the agentic vs. chaining distinction to show architectural understanding

---

## Template — Add Your Own

**Tags:** [comma-separated keywords for retrieval]
**Roles demonstrated:** [what PM skills does this show?]

### Situation
[Context — what was happening?]

### Task
[Your specific responsibility]

### Action
[What YOU did — decisions, influence, technical work]

### Result
[Outcomes — numbers, lessons, impact]
