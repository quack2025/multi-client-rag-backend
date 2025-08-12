# core/intelligent_suggestions.py
"""
Intelligent Suggestions Engine for Multi-Client RAG System
"""

from typing import Dict, List, Any

class IntelligentSuggestionEngine:
    """Generate intelligent suggestions based on RAG responses"""
    
    def __init__(self):
        print("[SUCCESS] Intelligent Suggestion Engine initialized")
    
    def analyze_response(self, 
                        answer: str, 
                        citations: List[Dict[str, Any]], 
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze response and generate suggestions"""
        try:
            suggestions = {
                "has_suggestions": True,
                "suggestions": [],
                "follow_up_questions": [],
                "related_topics": [],
                "data_gaps": []
            }
            
            client = metadata.get("client", "Cliente")
            chunks_retrieved = metadata.get("chunks_retrieved", 0)
            
            # Analyze citation quality
            if chunks_retrieved < 3:
                suggestions["data_gaps"].append(f"Pocos documentos encontrados ({chunks_retrieved}). Considera ampliar los criterios de búsqueda.")
            
            # Generate follow-up questions
            if "comparación" in answer.lower():
                suggestions["follow_up_questions"].extend([
                    f"¿Qué factores influyen en estas diferencias para {client}?",
                    "¿Cuáles son las implicaciones estratégicas de estos hallazgos?"
                ])
            
            if "tendencia" in answer.lower() or "evolución" in answer.lower():
                suggestions["follow_up_questions"].extend([
                    "¿Qué factores están impulsando esta tendencia?",
                    "¿Cómo podemos aprovechar esta evolución del mercado?"
                ])
            
            # Suggest related topics based on citations
            study_types = set()
            years = set()
            for citation in citations:
                if citation.get("study_type") and citation["study_type"] != "Unknown":
                    study_types.add(citation["study_type"])
                if citation.get("year") and citation["year"] != "Unknown":
                    years.add(str(citation["year"]))
            
            if len(study_types) > 1:
                suggestions["related_topics"].append("Análisis comparativo entre tipos de estudio disponibles")
            
            if len(years) > 1:
                suggestions["related_topics"].append("Análisis temporal de la evolución de los datos")
            
            # General suggestions
            suggestions["suggestions"].extend([
                "Exportar estos resultados para análisis adicional",
                "Filtrar por período específico para mayor precisión",
                f"Explorar datos relacionados de {client}"
            ])
            
            return suggestions
            
        except Exception as e:
            print(f"[ERROR] Error analyzing response: {e}")
            return {
                "has_suggestions": False,
                "error": str(e)
            }
    
    def generate_suggestion_text(self, suggestions: Dict[str, Any]) -> str:
        """Generate formatted suggestion text"""
        if not suggestions.get("has_suggestions", False):
            return ""
        
        suggestion_parts = []
        
        if suggestions.get("follow_up_questions"):
            suggestion_parts.append("\n\n💡 **Preguntas de seguimiento sugeridas:**")
            for question in suggestions["follow_up_questions"][:3]:
                suggestion_parts.append(f"• {question}")
        
        if suggestions.get("related_topics"):
            suggestion_parts.append("\n\n🔗 **Temas relacionados que podrías explorar:**")
            for topic in suggestions["related_topics"][:3]:
                suggestion_parts.append(f"• {topic}")
        
        if suggestions.get("data_gaps"):
            suggestion_parts.append("\n\n[WARNING] **Consideraciones sobre los datos:**")
            for gap in suggestions["data_gaps"]:
                suggestion_parts.append(f"• {gap}")
        
        return "\n".join(suggestion_parts)