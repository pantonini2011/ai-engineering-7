"""Borra un proyecto de Phoenix (sesión de pruebas descartada) para no mezclar métricas."""
import sys
from phoenix.client import Client
Client().projects.delete(project_name=sys.argv[1])
print("borrado:", sys.argv[1])
