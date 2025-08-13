# core/data_exporter.py
"""
Data Export functionality for Multi-Client RAG System
"""

import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import base64
from io import BytesIO

class RAGDataExporter:
    """Export RAG response data to multiple formats"""
    
    def __init__(self):
        print("[SUCCESS] RAG Data Exporter initialized")
    
    def export_rag_response(self, 
                           rag_response: Dict[str, Any], 
                           format_type: str = "excel",
                           include_metadata: bool = True) -> Dict[str, Any]:
        """Export RAG response to specified format"""
        try:
            if format_type.lower() == "excel":
                return self._export_to_excel(rag_response, include_metadata)
            elif format_type.lower() == "csv":
                return self._export_to_csv(rag_response, include_metadata)
            elif format_type.lower() == "json":
                return self._export_to_json(rag_response, include_metadata)
            elif format_type.lower() == "html":
                return self._export_to_html(rag_response, include_metadata)
            else:
                return {"error": f"Unsupported format: {format_type}"}
                
        except Exception as e:
            return {"error": f"Export failed: {str(e)}"}
    
    def _export_to_excel(self, rag_response: Dict[str, Any], include_metadata: bool) -> Dict[str, Any]:
        """Export to Excel format"""
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main answer sheet
                answer_df = pd.DataFrame([{
                    'Respuesta': rag_response.get('answer', ''),
                    'Timestamp': rag_response.get('timestamp', datetime.now().isoformat()),
                    'Cliente': rag_response.get('metadata', {}).get('client_name', 'N/A')
                }])
                answer_df.to_excel(writer, sheet_name='Respuesta', index=False)
                
                # Citations sheet
                citations = rag_response.get('citations', [])
                if citations:
                    citations_df = pd.DataFrame(citations)
                    citations_df.to_excel(writer, sheet_name='Citas', index=False)
                
                # Metadata sheet
                if include_metadata and rag_response.get('metadata'):
                    metadata_items = []
                    for key, value in rag_response['metadata'].items():
                        metadata_items.append({'Campo': key, 'Valor': str(value)})
                    metadata_df = pd.DataFrame(metadata_items)
                    metadata_df.to_excel(writer, sheet_name='Metadatos', index=False)
            
            output.seek(0)
            excel_data = base64.b64encode(output.read()).decode()
            
            return {
                "format": "excel",
                "data": excel_data,
                "filename": f"rag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
            
        except Exception as e:
            return {"error": f"Excel export failed: {str(e)}"}
    
    def _export_to_csv(self, rag_response: Dict[str, Any], include_metadata: bool) -> Dict[str, Any]:
        """Export to CSV format"""
        try:
            # Create main data
            data = {
                'Respuesta': [rag_response.get('answer', '')],
                'Timestamp': [rag_response.get('timestamp', datetime.now().isoformat())],
                'Cliente': [rag_response.get('metadata', {}).get('client_name', 'N/A')],
                'Modo': [rag_response.get('metadata', {}).get('mode', 'N/A')],
                'Chunks_Recuperados': [rag_response.get('metadata', {}).get('chunks_retrieved', 0)]
            }
            
            df = pd.DataFrame(data)
            csv_data = df.to_csv(index=False)
            csv_encoded = base64.b64encode(csv_data.encode()).decode()
            
            return {
                "format": "csv",
                "data": csv_encoded,
                "filename": f"rag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "mime_type": "text/csv"
            }
            
        except Exception as e:
            return {"error": f"CSV export failed: {str(e)}"}
    
    def _export_to_json(self, rag_response: Dict[str, Any], include_metadata: bool) -> Dict[str, Any]:
        """Export to JSON format"""
        try:
            export_data = rag_response.copy()
            
            if not include_metadata and 'metadata' in export_data:
                del export_data['metadata']
            
            # Add export info
            export_data['export_info'] = {
                'exported_at': datetime.now().isoformat(),
                'format': 'json',
                'include_metadata': include_metadata
            }
            
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            json_encoded = base64.b64encode(json_data.encode()).decode()
            
            return {
                "format": "json",
                "data": json_encoded,
                "filename": f"rag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "mime_type": "application/json"
            }
            
        except Exception as e:
            return {"error": f"JSON export failed: {str(e)}"}
    
    def _export_to_html(self, rag_response: Dict[str, Any], include_metadata: bool) -> Dict[str, Any]:
        """Export to HTML format"""
        try:
            client_name = rag_response.get('metadata', {}).get('client_name', 'Cliente')
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Reporte RAG - {client_name}</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                    .answer {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                    .citations {{ margin: 20px 0; }}
                    .citation {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                    .metadata {{ font-size: 0.9em; color: #666; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Reporte RAG - {client_name}</h1>
                    <p>Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </div>
                
                <div class="answer">
                    <h2>Respuesta</h2>
                    <p>{rag_response.get('answer', '').replace(chr(10), '<br>')}</p>
                </div>
            """
            
            # Add citations if available
            citations = rag_response.get('citations', [])
            if citations:
                html_content += """
                <div class="citations">
                    <h2>Fuentes</h2>
                    <table>
                        <tr>
                            <th>Documento</th>
                            <th>Tipo de Estudio</th>
                            <th>Año</th>
                            <th>Relevancia</th>
                        </tr>
                """
                
                for citation in citations:
                    html_content += f"""
                        <tr>
                            <td>{citation.get('document', 'N/A')}</td>
                            <td>{citation.get('study_type', 'N/A')}</td>
                            <td>{citation.get('year', 'N/A')}</td>
                            <td>{citation.get('similarity', 'N/A')}</td>
                        </tr>
                    """
                
                html_content += "</table></div>"
            
            # Add metadata if requested
            if include_metadata and rag_response.get('metadata'):
                html_content += """
                <div class="metadata">
                    <h2>Metadatos</h2>
                    <table>
                        <tr><th>Campo</th><th>Valor</th></tr>
                """
                
                for key, value in rag_response['metadata'].items():
                    html_content += f"<tr><td>{key}</td><td>{str(value)}</td></tr>"
                
                html_content += "</table></div>"
            
            html_content += "</body></html>"
            
            html_encoded = base64.b64encode(html_content.encode()).decode()
            
            return {
                "format": "html",
                "data": html_encoded,
                "filename": f"rag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "mime_type": "text/html"
            }
            
        except Exception as e:
            return {"error": f"HTML export failed: {str(e)}"}