from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LeadInsight:
    """Individual insight about the lead."""
    category: str  # e.g., "machine_interest", "budget", "urgency", "business_type"
    description: str
    confidence: float  # 0.0 to 1.0


@dataclass
class LeadQualification:
    """Structured lead qualification output."""
    session_id: str
    phone_number: str
    lead_tier: str  # "cold", "warm", "hot"
    priority_score: float  # 0 to 100, higher = more priority
    priority_level: str  # "low", "medium", "high"
    estimated_budget: Optional[str]  # e.g., "INR 50,000 - 2,00,000"
    machine_interests: list[str]  # e.g., ["CNC Router", "Laser Cutter"]
    business_type: Optional[str]  # e.g., "Woodworking Shop", "Sign Making"
    urgency_level: str  # "low", "medium", "high"
    contact_preference: Optional[str]  # e.g., "WhatsApp", "Call"
    insights: list[LeadInsight]
    recommended_next_steps: list[str]
    analysis_timestamp: str
    total_messages_exchanged: int
    conversation_duration_minutes: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['insights'] = [asdict(i) for i in self.insights]
        return data


def build_lead_analysis_prompt(chat_history: list[dict]) -> str:
    """
    Build a comprehensive prompt for analyzing chat history and qualifying leads.
    
    Args:
        chat_history: List of chat messages with timestamps and roles.
    
    Returns:
        A formatted prompt string for the LLM.
    """
    
    # Format chat history for readability
    formatted_history = "\n".join(
        [f"[{msg.get('timestamp', 'N/A')}] {msg.get('role', 'unknown').upper()}: {msg.get('content', '')}" 
         for msg in chat_history]
    )
    
    prompt = f"""You are an expert B2B sales lead qualification analyst for Woodmaster CNC, a supplier of professional CNC machines and tools.

Analyze the following chat conversation and extract lead qualification data. Your analysis should be insightful, data-driven, and help prioritize follow-up actions.

===== CHAT HISTORY =====
{formatted_history}

===== ANALYSIS TASK =====
Based on the above conversation, provide a JSON response with the following structure:

{{
    "lead_tier": "<cold|warm|hot>",
    "priority_score": <0-100>,
    "priority_level": "<low|medium|high>",
    "estimated_budget": "<budget range or null if not mentioned>",
    "machine_interests": ["<machine type 1>", "<machine type 2>"],
    "business_type": "<type of business or null>",
    "urgency_level": "<low|medium|high>",
    "contact_preference": "<WhatsApp|Call|Email or null>",
    "insights": [
        {{
            "category": "<category name>",
            "description": "<detailed insight>",
            "confidence": <0.0-1.0>
        }}
    ],
    "recommended_next_steps": ["<action 1>", "<action 2>", "<action 3>"]
}}

===== CLASSIFICATION CRITERIA =====

### LEAD TIER Classification:

**HOT LEAD** (Immediate Sales Potential):
- Explicitly asks for pricing, quotes, or product demo
- Mentions a specific machine model or detailed specifications
- Indicates immediate need (next 1-4 weeks)
- Shows high engagement (5+ messages, detailed questions)
- Mentions budget or purchase timeline
- Asks about delivery, warranty, or technical support
- Company name or business details mentioned

**WARM LEAD** (Sales Development Potential):
- Shows genuine interest (multiple relevant questions)
- Asks about capabilities or machine features
- Mentions future plans (3-6 months timeline)
- Moderate engagement (3-4 messages)
- Implies potential budget or capacity
- Asks about training, support, or implementation

**COLD LEAD** (Awareness/Nurture Stage):
- Generic or curiosity-driven questions
- No specific machine interest expressed
- No timeline or budget indicators
- Low engagement (1-2 messages)
- Comparison shopping mode
- Testing the bot/service

### PRIORITY_SCORE (0-100):
Calculate based on:
- Lead tier weight: Hot=60-100, Warm=30-60, Cold=0-30
- Engagement level (+10-20 points for 5+ messages, detailed questions)
- Specificity of need (+15 points for named machines, models)
- Budget clarity (+15 points for mentioned or implied budget)
- Timeline urgency (+20 points for immediate or very soon)
- Business indicator (+10 points for company/business mention)

### MACHINE_INTERESTS:
Extract specific CNC machines or capabilities mentioned:
- CNC Router, CNC Engraving Machine, Laser Cutter, Wood Lathe, Spindle, Bits/Tools
- Also consider implied interests from use cases (e.g., "sign making" → CNC Router)

### BUSINESS_TYPE:
Classify the type of business:
- Woodworking Shop, Furniture Maker, Sign Making, Craft/Hobby, Industrial Manufacturing,
- Engineering Firm, Educational Institution, or Other

### URGENCY_LEVEL:
- HIGH: "ASAP", "immediately", "next week", specific timeline mentioned, repeated urgency
- MEDIUM: "next month", "3-6 months", exploratory but serious
- LOW: "exploring options", "maybe in future", unclear timeline

### INSIGHTS Generation:
Generate 3-5 actionable insights under categories like:
- machine_interest: What specific machine or capability caught their attention
- business_application: How they plan to use the machine
- budget_indicators: Clues about their financial capacity
- urgency_indicators: Time sensitivity of their need
- engagement_quality: How engaged they are in the conversation
- technical_readiness: Are they technically prepared or do they need support?
- decision_maker: Are they the decision maker or need approval?

### RECOMMENDED_NEXT_STEPS:
Suggest 2-4 specific follow-up actions based on lead tier and needs:
- For HOT: "Send detailed product specs and pricing quote", "Schedule demo call", "Confirm delivery timeline"
- For WARM: "Send product catalog and case studies", "Share testimonials from similar businesses", "Offer free consultation"
- For COLD: "Add to nurture sequence", "Share industry insights/blog", "Follow up in 2 weeks"

===== IMPORTANT GUIDELINES =====
1. Be conservative in tier assignment - only mark as HOT if truly sales-ready
2. Consider the tone and engagement level in the conversation
3. Extract insights from both explicit statements and implicit signals (e.g., type of questions asked)
4. Provide actionable recommendations that align with each lead tier
5. Return ONLY valid JSON, no additional text or commentary
6. If a field cannot be determined from the conversation, use null

Return ONLY the JSON object, no markdown formatting or explanations."""
    
    return prompt


def parse_lead_analysis_response(
    response_text: str,
    session_id: str,
    phone_number: str,
    chat_history: list[dict]
) -> LeadQualification | None:
    """
    Parse the LLM response and create a LeadQualification object.
    
    Args:
        response_text: The LLM's JSON response
        session_id: The session identifier
        phone_number: The contact phone number
        chat_history: The original chat messages
    
    Returns:
        LeadQualification object or None if parsing fails
    """
    try:
        # Extract JSON from response (handle potential markdown or extra text)
        json_str = response_text.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        
        data = json.loads(json_str)
        
        # Parse insights
        insights = [
            LeadInsight(
                category=i.get("category", "unknown"),
                description=i.get("description", ""),
                confidence=float(i.get("confidence", 0.5))
            )
            for i in data.get("insights", [])
        ]
        
        # Calculate conversation metrics
        total_messages = len(chat_history)
        if len(chat_history) > 1:
            first_timestamp = chat_history[0].get("timestamp")
            last_timestamp = chat_history[-1].get("timestamp")
            # Simplified: assume each message is ~2 minutes apart
            conversation_duration = total_messages * 2
        else:
            conversation_duration = 1
        
        qualification = LeadQualification(
            session_id=session_id,
            phone_number=phone_number,
            lead_tier=data.get("lead_tier", "cold").lower(),
            priority_score=float(data.get("priority_score", 0)),
            priority_level=data.get("priority_level", "low").lower(),
            estimated_budget=data.get("estimated_budget"),
            machine_interests=data.get("machine_interests", []),
            business_type=data.get("business_type"),
            urgency_level=data.get("urgency_level", "low").lower(),
            contact_preference=data.get("contact_preference"),
            insights=insights,
            recommended_next_steps=data.get("recommended_next_steps", []),
            analysis_timestamp=datetime.utcnow().isoformat() + "Z",
            total_messages_exchanged=total_messages,
            conversation_duration_minutes=conversation_duration,
        )
        
        return qualification
    
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Failed to parse lead analysis response: {e}\nResponse: {response_text}")
        return None


def create_dummy_lead_qualification(
    session_id: str,
    phone_number: str,
    chat_history: list[dict]
) -> LeadQualification:
    """
    Create a dummy lead qualification for testing (when LLM is unavailable).
    
    Args:
        session_id: Session ID
        phone_number: Phone number
        chat_history: Chat messages
    
    Returns:
        A dummy LeadQualification object
    """
    
    # Simple heuristic for demo purposes
    total_messages = len(chat_history)
    
    if total_messages >= 5:
        tier = "warm"
        score = 50
    elif total_messages >= 2:
        tier = "warm"
        score = 35
    else:
        tier = "cold"
        score = 15
    
    return LeadQualification(
        session_id=session_id,
        phone_number=phone_number,
        lead_tier=tier,
        priority_score=score,
        priority_level="high" if score >= 60 else "medium" if score >= 30 else "low",
        estimated_budget=None,
        machine_interests=["CNC Router"],
        business_type=None,
        urgency_level="medium",
        contact_preference="WhatsApp",
        insights=[
            LeadInsight(
                category="engagement_quality",
                description=f"User sent {total_messages} messages in this conversation",
                confidence=0.9
            )
        ],
        recommended_next_steps=["Send product catalog", "Schedule a demo call"],
        analysis_timestamp=datetime.utcnow().isoformat() + "Z",
        total_messages_exchanged=total_messages,
        conversation_duration_minutes=max(1, total_messages * 2),
    )
