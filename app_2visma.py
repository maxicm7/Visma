import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import re
import numpy as np
import csv
import traceback # Para depuración detallada si es necesario

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Análisis Operativo VISMA", initial_sidebar_state="expanded")

# --- Header ---
st.title("Análisis Operativo y Gestión de Costos VISMA")
st.sidebar.header("Carga de Archivos CSV")

# --- File Uploaders ---
stock_file = st.sidebar.file_uploader("Cargar Archivo de Stock (Ej: stock_simple.csv)", type=["csv", "txt"])
eco_file = st.sidebar.file_uploader("Cargar Archivo de Presupuesto (Ej: presupuesto_simple.csv)", type=["csv", "txt"])
fuel_file = st.sidebar.file_uploader("Cargar Archivo de Combustible (Ej: combustible_simple_v2.csv)", type=["csv", "txt"])


# --- Helper functions ---

def safe_float_parse(value_str):
    """
    Intenta convertir una cadena limpia a float.
    Retorna 0.0 si la entrada es vacía, '-', '.', '$', None, NaN, u otros errores comunes,
    o si falla la conversión después de una limpieza básica.
    Maneja formatos comunes de decimales (. o ,).
    """
    if pd.isna(value_str) or str(value_str).strip() in ['', '-', '.', '$', 'nan', '#N/A', 'N/A', '#VALUE!', 'None']:
        return 0.0
    try:
        value_str = str(value_str).strip().replace('$', '').replace(' ', '') # Basic cleaning

        # Handle potential thousands separators and decimal comma/dot
        # If comma exists and dot exists, and comma is last, treat comma as decimal
        if ',' in value_str and '.' in value_str:
            last_dot = value_str.rfind('.')
            last_comma = value_str.rfind(',')
            if last_comma > last_dot: # Comma is likely decimal
                value_str = value_str.replace('.', '').replace(',', '.')
            else: # Dot is likely decimal, comma is thousands
                value_str = value_str.replace(',', '')
        elif ',' in value_str: # Only comma, likely decimal
            value_str = value_str.replace(',', '.')

        if not value_str or value_str in ['+', '-']: return 0.0 # Final check after cleaning

        return float(value_str)

    except ValueError:
        # print(f"DEBUG: ValueError parsing '{value_str}'") # Optional: uncomment for debugging
        return 0.0
    except Exception as e: # Catch generic exceptions too
        # print(f"DEBUG: Generic Error parsing '{value_str}': {e}") # Optional: uncomment for debugging
        return 0.0


def find_header_row(content_lines, expected_headers_list, search_range=50):
    """
    Busca una línea que contenga exactamente los nombres de columna esperados
    en las primeras `search_range` líneas.
    Maneja BOM (Byte Order Mark) al inicio y usa csv.reader para robustez con comas embebidas.
    Retorna el índice (0-based) de la línea si se encuentra, y los nombres limpios encontrados,
    o -1 y None si no se encuentra.
    """
    for i, line in enumerate(content_lines[:search_range]):
        line_strip = line.strip()
        if not line_strip: continue

        try:
            # Use csv reader to correctly split fields, especially if commas are embedded and fields are quoted
            # Using a basic delimiter=',' assumes the main separator is comma.
            reader = csv.reader(io.StringIO(line_strip), delimiter=',')
            header_fields = next(reader) # Read the first (and only) row

            # Clean field names: strip whitespace, handle potential BOM
            cleaned_fields = [f.strip() for f in header_fields]
            if cleaned_fields and cleaned_fields[0].startswith('\ufeff'):
                cleaned_fields[0] = cleaned_fields[0][1:] # Remove BOM

            # Check for exact match
            if cleaned_fields == expected_headers_list:
                return i, cleaned_fields

        except csv.Error:
            # print(f"DEBUG: csv.Error reading potential header line {i+1}: {line_strip[:80]}") # Optional: uncomment
            pass # Ignore lines that don't parse as simple CSV
        except Exception:
            # print(f"DEBUG: find_header_row encountered other error on line {i+1}: {e} - {line_strip[:80]}") # Optional: uncomment
            pass

    # Header not found in the search range
    return -1, None


# --- Parsing Functions ---

def parse_stock_visma(uploaded_file):
    """
    Parsea el archivo de Stock con la estructura simple (Codigo, Producto, Categoria,
    CantidadActual, CostoUnitario, Ubicacion, STOCK_MINIMO).
    Busca una cabecera exacta. Calcula Valor Total Item.
    Maneja la posible falta de la columna STOCK_MINIMO para compatibilidad.
    Returns: pd.DataFrame on success (can be empty but structured), None on critical error
    """
    try:
        # Read file content line by line
        content = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors='replace')).readlines() # Use errors='replace' for potential decoding issues

        # --- Define the EXACT expected headers for the simple stock file ---
        expected_header_full = ['Codigo', 'Producto', 'Categoria', 'CantidadActual', 'CostoUnitario', 'Ubicacion', 'STOCK_MINIMO']
        expected_header_no_min = ['Codigo', 'Producto', 'Categoria', 'CantidadActual', 'CostoUnitario', 'Ubicacion'] # Fallback

        # --- Find the exact header row, trying full first, then fallback ---
        header_row_index, detected_header_names = find_header_row(content, expected_header_full)

        use_stock_min_column = (header_row_index != -1) # Assume we can use STOCK_MINIMO if full header found

        if header_row_index == -1:
            # Try the fallback header without STOCK_MINIMO
            header_row_index, detected_header_names = find_header_row(content, expected_header_no_min)
            if header_row_index == -1:
                 # Neither header found, show error and return None
                 st.error(f"Stock parse (Simple): No se encontró una fila de cabecera que coincida exactamente con '{', '.join(expected_header_full)}' o '{', '.join(expected_header_no_min)}' en las primeras 50 líneas.")
                 st.write("Primeras líneas leídas (muestra):")
                 sample_lines = [line.strip() for line in content[:min(len(content), 15)] if line.strip()]
                 st.text("\n".join(sample_lines) if sample_lines else "Archivo vacío o ilegible.")
                 return None # Cannot parse data without a recognizable header
            else:
                 # Found the header without STOCK_MINIMO
                 st.warning("Stock parse (Simple): Cabecera 'STOCK_MINIMO' no encontrada. El análisis de stock mínimo no estará disponible.")
                 use_stock_min_column = False # Confirm we are NOT using the min column


        # --- Read all rows after the header using csv.reader for robustness ---
        data_rows = []
        # Determine the expected number of fields based on the detected header
        expected_field_count = len(detected_header_names)

        for i in range(header_row_index + 1, len(content)):
             line = content[i].strip()
             if not line: continue # Skip empty lines

             try:
                  reader = csv.reader(io.StringIO(line), delimiter=',')
                  fields = next(reader)

                  # Pad short rows with empty strings to match the number of header fields.
                  # This is generally safer than skipping if the row just has missing trailing data.
                  if len(fields) < expected_field_count:
                       padded_fields = fields + [''] * (expected_field_count - len(fields))
                       # Optionally warn: st.warning(f"Stock parse (Simple): Línea {i+1} rellenada de {len(fields)} a {expected_field_count} campos.")
                       data_rows.append(padded_fields)
                  else:
                       # Take only the first expected_field_count fields from long rows
                       data_rows.append(fields[:expected_field_count])


             except csv.Error:
                  # print(f"DEBUG: csv.Error reading data line {i+1}: {line[:80]}") # Optional: uncomment
                  st.warning(f"Stock parse (Simple): Línea de datos no parseable como CSV encontrada y omitida (línea {i+1}): {line[:80]}...")
                  pass
             except Exception as e:
                   # print(f"DEBUG: Stock data parse generic error line {i+1}: {e} - {line[:80]}") # Optional: uncomment
                   st.warning(f"Stock parse (Simple): Error procesando línea {i+1}, omitiendo: {line[:80]}... Error: {e}")
                   pass


        # Create a pandas DataFrame using the detected headers
        # Define the list of ALL columns we want in the final output DataFrame regardless of input
        final_cols_order_template = ['Codigo', 'Producto', 'Categoria', 'CantidadActual', 'CostoUnitario', 'Valor Total Item', 'Ubicacion', 'STOCK_MINIMO']

        if not data_rows:
             st.warning("Stock parse (Simple): Archivo procesado, cabecera encontrada, pero no hay filas de datos válidas.")
             # Return an empty DF with ALL expected final columns for structure consistency
             return pd.DataFrame(columns=final_cols_order_template)


        # Create DF from raw data rows with detected headers
        df_raw_data = pd.DataFrame(data_rows, columns=detected_header_names)


        # --- Data Cleaning and Type Conversion ---
        df_cleaned = df_raw_data.copy() # Work on a copy


        # Convert known string columns and handle potential NaNs/None, strip whitespace
        # Use the actual column names from df_cleaned before attempting conversion
        string_cols_present = [col for col in ['Codigo', 'Producto', 'Categoria', 'Ubicacion'] if col in df_cleaned.columns]
        for col in string_cols_present:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().fillna('')

        # Add missing expected string cols with empty string default if not present
        for col in ['Codigo', 'Producto', 'Categoria', 'Ubicacion']:
             if col not in df_cleaned.columns:
                  df_cleaned[col] = ''


        # Convert numeric columns using safe_float_parse.
        # Use the actual column names from df_cleaned before attempting conversion
        numeric_cols_present = [col for col in ['CantidadActual', 'CostoUnitario'] if col in df_cleaned.columns]
        for col in numeric_cols_present:
             # Apply safe_float_parse row by row for robustness
             df_cleaned[col] = df_cleaned[col].apply(safe_float_parse)

        # Handle STOCK_MINIMO specifically based on whether it was detected
        if 'STOCK_MINIMO' in df_cleaned.columns:
             df_cleaned['STOCK_MINIMO'] = df_cleaned['STOCK_MINIMO'].apply(safe_float_parse)
        else:
             # Add STOCK_MINIMO with default 0.0 if it wasn't in the header
             df_cleaned['STOCK_MINIMO'] = 0.0


        # Add any other missing numeric columns with default 0.0 if not present (shouldn't be needed based on expected headers)
        for col in ['CantidadActual', 'CostoUnitario']:
             if col not in df_cleaned.columns:
                  df_cleaned[col] = 0.0


        # --- Calculate Derived Columns ---
        # Ensure base columns are numeric before calculation
        df_cleaned['Valor Total Item'] = df_cleaned['CantidadActual'].fillna(0.0) * df_cleaned['CostoUnitario'].fillna(0.0)


        # --- Final DataFrame Selection and Order ---
        # Define the list of ALL columns we want in the final output DataFrame
        final_cols_order = ['Codigo', 'Producto', 'Categoria', 'CantidadActual', 'CostoUnitario', 'Valor Total Item', 'Ubicacion', 'STOCK_MINIMO']

        # Ensure all final columns exist in df_cleaned, adding them with defaults if necessary
        # This loop is partially redundant with the cleaning steps above but ensures all final columns are present just before filtering/ordering
        for col in final_cols_order:
             if col not in df_cleaned.columns:
                  if col in ['CantidadActual', 'CostoUnitario', 'Valor Total Item', 'STOCK_MINIMO']:
                       df_cleaned[col] = 0.0
                  else:
                       df_cleaned[col] = ''


        # Filter out rows where 'Codigo' or 'Categoria' are empty/whitespace *after* cleanup
        # This ensures only valid inventory items are kept. Check after adding default columns.
        if 'Codigo' in df_cleaned.columns and 'Categoria' in df_cleaned.columns:
             df_cleaned = df_cleaned[
                 (df_cleaned['Codigo'] != '') &
                 (df_cleaned['Categoria'] != '')
             ].copy()
        else: # This state should ideally not happen if checks above added missing, but safety
             st.error("Stock parse (Simple): Columnas 'Codigo' o 'Categoria' inesperadamente faltantes después de procesamiento. No se puede filtrar items inválidos.")


        # Select and reorder columns for the final DataFrame
        # Use .loc to ensure order and column presence safety after filtering
        df_final = df_cleaned.loc[:, [col for col in final_cols_order if col in df_cleaned.columns]]


        return df_final # Return the DataFrame if successful

    except Exception as e:
        st.error(f"Stock parse (Simple): Error general inesperado durante el procesamiento del archivo: {e}")
        # st.error(traceback.format_exc()) # Uncomment for detailed debugging in console/logs
        # On general parse failure, return None as originally intended for critical errors
        return None


def parse_eco_visma(uploaded_file):
    """
    Parsea el archivo de Presupuesto con la estructura simple (Cabecera: Categoria, Invierno, Verano;
    Filas: key, valorInv, valorVer). Mapea las filas de categoría a las claves internas de la app.
    Espera una cabecera exacta y 3 campos por fila de datos.
    Returns: dictionary on success (can be empty), None on critical error
    """
    try:
        content = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors='replace')).readlines()

        # --- Find the exact header row for the simple structure ---
        expected_header_simple = ['Categoria', 'Invierno', 'Verano']
        header_row_index, detected_header_names = find_header_row(content, expected_header_simple)

        # Define the structure that should be returned even if parsing fails critically
        # This structure is also used as the default zeroed structure
        expected_internal_keys_eco = ['Movilizacion', 'Costos Directos', 'Costos Indirectos/Generales', 'Utilidades']
        zeroed_eco_data_structure = {
           'invierno': {key: 0.0 for key in expected_internal_keys_eco},
           'verano': {key: 0.0 for key in expected_internal_keys_eco}
        }


        if header_row_index == -1:
            st.error(f"ECO parse (Simple): No se encontró la fila de cabecera exacta '{', '.join(expected_header_simple)}' en las primeras 50 líneas del archivo.")
            st.write("Primeras líneas leídas (muestra):")
            sample_lines = [line.strip() for line in content[:min(len(content), 15)] if line.strip()]
            st.text("\n".join(sample_lines) if sample_lines else "Archivo vacío o ilegible.")
            st.warning("Verifica que estás subiendo el archivo de PRESUPUESTO SIMPLE (no el de Stock o Combustible).")
            return None # Return None if header isn't found

        processed_header_names = detected_header_names
        expected_field_count = len(processed_header_names)

        # --- Read all data rows after the header using csv.reader ---
        data_rows = []
        for i in range(header_row_index + 1, len(content)):
             line = content[i].strip()
             if not line: continue # Skip empty lines

             try:
                  reader = csv.reader(io.StringIO(line), delimiter=',')
                  fields = next(reader)

                  # Strict check for expected number of fields for this simple structure
                  if len(fields) != expected_field_count:
                       st.warning(f"ECO parse (Simple): Línea de datos con número incorrecto de campos encontrada y omitida (línea {i+1}, campos: {len(fields)}, esperados: {expected_field_count}): {line[:80]}...")
                       continue # Skip invalid data rows

                  data_rows.append(fields)

             except csv.Error:
                  # print(f"DEBUG: csv.Error reading ECO data line {i+1}: {line[:80]}") # Optional: uncomment
                  st.warning(f"ECO parse (Simple): Línea de datos no parseable como CSV encontrada y omitida (línea {i+1}): {line[:80]}...")
                  pass
             except Exception as e:
                   # print(f"DEBUG: ECO data parse generic error line {i+1}: {e} - {line[:80]}") # Optional: uncomment
                   st.warning(f"ECO parse (Simple): Error procesando línea {i+1}, omitiendo: {line[:80]}... Error: {e}")
                   pass

        # Create a pandas DataFrame from raw data rows
        if not data_rows:
             st.warning("ECO parse (Simple): Archivo procesado, cabecera encontrada, pero no hay filas de datos válidas con 3 campos.")
             # Return the zeroed structure even if no data rows are found
             return zeroed_eco_data_structure

        df_raw_data = pd.DataFrame(data_rows, columns=processed_header_names)

        # --- Data Cleaning and Mapeo to Internal Structure ---
        df_cleaned = df_raw_data.copy()

        # Ensure Categoria column exists and is string, strip whitespace
        if 'Categoria' in df_cleaned.columns:
             df_cleaned['Categoria'] = df_cleaned['Categoria'].astype(str).str.strip()
        else:
             st.error("ECO parse (Simple): Columna 'Categoria' faltante después de leer los datos.")
             # Return the zeroed structure if critical column is missing
             return zeroed_eco_data_structure

        # Ensure numeric columns exist and apply safe_float_parse
        numeric_cols_eco = ['Invierno', 'Verano']
        for col in numeric_cols_eco:
             if col in df_cleaned.columns:
                  df_cleaned[col] = df_cleaned[col].apply(safe_float_parse)
             else:
                  st.warning(f"ECO parse (Simple): Columna '{col}' faltante. Usando 0.0 para estos valores.")
                  df_cleaned[col] = 0.0


        # This dictionary maps the 'Categoria' names expected in the simple CSV
        # to the internal keys used in the rest of the Streamlit app's logic and dictionaries.
        category_mapping_to_internal_key = {
            'Movilizacion': 'Movilizacion',
            'CostosDirectos': 'Costos Directos',
            'CostosIndirectosGenerales': 'Costos Indirectos/Generales',
            'Utilidades': 'Utilidades',
        }

        # Internal dictionary structure required by the rest of the app's plotting/display logic
        parsed_costs = {
            'invierno': {key: 0.0 for key in expected_internal_keys_eco},
            'verano': {key: 0.0 for key in expected_internal_keys_eco}
        }

        # Populate parsed_costs from cleaned DataFrame
        if not df_cleaned.empty:
            # Ensure Categoria, Invierno, Verano columns exist (checked above, but belt and suspenders)
            if all(col in df_cleaned.columns for col in ['Categoria', 'Invierno', 'Verano']):
                for index, row in df_cleaned.iterrows():
                    category_name_in_file = row['Categoria'] # Already stripped above

                    internal_category_key = category_mapping_to_internal_key.get(category_name_in_file)

                    if internal_category_key in expected_internal_keys_eco:
                        # Sum values if multiple rows match the same category key
                        parsed_costs['invierno'][internal_category_key] += row['Invierno']
                        parsed_costs['verano'][internal_category_key] += row['Verano']
                    # Optional: Warning for unrecognized categories removed to reduce noise,
                    # assuming only expected categories matter for the app's fixed structure.

            else:
                 st.error("ECO parse (Simple): Columnas 'Categoria', 'Invierno' o 'Verano' faltantes después de leer y limpiar los datos.")
                 # Return the zeroed structure even if essential columns are missing after cleaning
                 return zeroed_eco_data_structure

        return parsed_costs # Return the dictionary if successful

    except Exception as e:
        st.error(f"ECO parse (Simple): Error general inesperado durante el procesamiento del archivo: {e}")
        # st.error(traceback.format_exc()) # Uncomment for detailed debugging
        # On general parse failure, return None as originally intended for critical errors
        return None


def parse_fuel_log(uploaded_file):
    """
    Parsea el archivo de registro de combustible con la ESTRUCTURA SIMPLE V2.
    Espera líneas SETUP al inicio y luego una cabecera de datos simple
    seguida de filas de datos. Lee CostoUnitarioLt y calcula Costo Total Egreso.
    Convierte 'Fecha' a datetime y renombra columnas Lts/Equipo para coincidir con estándar.
    Returns: tuple (pd.DataFrame, dict) on success (DF can be empty but structured), None on critical error
    """
    # Define the structure that should be returned even if parsing fails critically
    empty_fuel_cols_template = ['Fecha', 'Equipo/Int.', 'Codigo', 'Lts Ingreso', 'Lts Egreso', 'HsKm', 'Comentarios', 'Tipo de Comb.', 'Hs/Km_Numeric', 'Costo Unitario Lt', 'Costo Total Egreso']
    zeroed_initial_stock_template = {'GASOIL': 0.0, 'NAFTA': 0.0}

    try:
        # Read file content line by line
        content = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors='replace')).readlines() # Use errors='replace'

        # --- Initialize storage for setup data and main data ---
        saldo_gasoil = 0.0
        saldo_nafta = 0.0
        fuel_type_mapping = {} # Maps CodigoCombustible to 'GASOIL', 'NAFTA', etc.
        data_header_index = -1 # Index of the line with the header for data rows

        # Define the EXACT expected header names for the data block in the new simple format V2
        expected_data_header_simple_v2 = ['Fecha', 'Equipo', 'CodigoCombustible', 'LtsIngreso', 'LtsEgreso', 'HsKm', 'Comentarios', 'CostoUnitarioLt']


        # --- Scan for SETUP lines and the DATA header line ---
        # Limit scan to first N lines to be efficient
        search_range_setup = min(len(content), 100) # Scan up to 100 lines or EOF
        for i, line in enumerate(content[:search_range_setup]):
             line_strip = line.strip()
             if not line_strip: continue

             try:
                  reader = csv.reader(io.StringIO(line_strip), delimiter=',')
                  line_fields = next(reader)

                  # Handle BOM in the very first field of the very first line
                  if i == 0 and line_fields and line_fields[0].startswith('\ufeff'):
                       line_fields[0] = line_fields[0][1:]

                  line_fields_clean = [f.strip() for f in line_fields]

             except csv.Error:
                   # print(f"DEBUG: csv.Error during SETUP scan line {i+1}: {line_strip[:80]}") # Optional: uncomment
                   continue # Skip lines that don't parse as simple CSV during setup scan
             except Exception as e:
                  # print(f"DEBUG: Generic error during SETUP scan line {i+1}: {e} - {line_strip[:80]}") # Optional: uncomment
                  continue # Skip on other unexpected errors during setup scan


             if line_fields_clean and line_fields_clean[0].upper() == 'SETUP':
                 if len(line_fields_clean) > 1:
                      setup_type = line_fields_clean[1].upper()

                      if setup_type == 'SALDO INICIAL GASOIL':
                           if len(line_fields_clean) > 2:
                                saldo_gasoil = safe_float_parse(line_fields_clean[2]) # Parse saldo

                      elif setup_type == 'SALDO INICIAL NAFTA':
                           if len(line_fields_clean) > 2:
                                saldo_nafta = safe_float_parse(line_fields_clean[2]) # Parse saldo

                      elif setup_type == 'MAPEO CODIGO':
                           if len(line_fields_clean) > 3:
                                code = line_fields_clean[2]
                                fuel_type = line_fields_clean[3] # Keep original case of fuel type from file
                                if code and fuel_type:
                                   fuel_type_mapping[code.strip()] = fuel_type.strip() # Store stripped values


             # Check if THIS line EXACTLY matches the expected data header list
             # Ensure we haven't already found the header (in case it appears multiple times)
             elif data_header_index == -1 and line_fields_clean == expected_data_header_simple_v2:
                 data_header_index = i
                 break # Found the data header, stop scanning setup lines


        # --- Validate if DATA header was found ---
        if data_header_index == -1:
            st.error(f"Fuel parse (Simple V2): No se encontró la fila de encabezado de datos exacta '{', '.join(expected_data_header_simple_v2)}' en las primeras {search_range_setup} líneas.")
            st.write("Las primeras líneas encontradas al intentar leer SETUP y cabeceras fueron (muestra):")
            sample_lines_read = [line.strip() for line in content[:min(len(content), 20)] if line.strip()]
            st.text("\n".join(sample_lines_read) if sample_lines_read else "Archivo vacío o ilegible.")
            st.warning("Verifica que estás subiendo el archivo de COMBUSTIBLE SIMPLE V2 y que tiene la cabecera de datos exacta (con todos los nombres y en el orden correcto).")
            # Return empty DF with expected structure and initial stock found so far
            return pd.DataFrame(columns=empty_fuel_cols_template), {'GASOIL': saldo_gasoil, 'NAFTA': saldo_nafta}


        # --- Read the Data block using pandas starting from the header ---
        data_block_string = "".join(content[data_header_index:])
        df_raw = pd.DataFrame() # Initialize empty DF in case read_csv fails

        try:
            # Read as CSV, using the identified header row, reading everything as string initially
            # Use header=0 to indicate the first row of the data_block_string is the header
            df_raw = pd.read_csv(io.StringIO(data_block_string), sep=',', header=0, dtype=str, keep_default_na=False, low_memory=False)
        except Exception as e:
            st.error(f"Fuel parse (Simple V2): Error leyendo el bloque de datos CSV con cabecera (desde línea {data_header_index + 1}): {e}. Revisa la consistencia de formato entre filas de datos debajo de la cabecera.")
            # st.error(traceback.format_exc()) # Uncomment for detailed debugging
            # Return empty DF with expected structure and initial stock found so far
            return pd.DataFrame(columns=empty_fuel_cols_template), {'GASOIL': saldo_gasoil, 'NAFTA': saldo_nafta}


        # --- Data Cleaning and Type Conversion ---
        df = df_raw.copy()

        # If df_raw is empty or has no columns after reading, return empty structured DF
        if df.empty or len(df.columns) == 0:
             st.warning("Fuel parse (Simple V2): El archivo fue leído, cabecera encontrada, pero no se extrajeron columnas o datos.")
             return pd.DataFrame(columns=empty_fuel_cols_template), {'GASOIL': saldo_gasoil, 'NAFTA': saldo_nafta}


        # Rename/Align columns based on expected header names.
        # Assume the columns read by pandas correspond positionally to the expected header if the count matches.
        # If count doesn't match, pandas read something unexpected.
        if len(df.columns) == len(expected_data_header_simple_v2):
             column_name_mapping = dict(zip(df.columns, expected_data_header_simple_v2))
             df.rename(columns=column_name_mapping, inplace=True)
        else:
             st.warning(f"Fuel parse (Simple V2): El número de columnas leídas por pandas ({len(df.columns)}) no coincide con la cabecera esperada ({len(expected_data_header_simple_v2)}). Revisa el delimitador. Intentando procesar con columnas disponibles.")
             # Attempt to proceed, relying on later steps to add missing standard columns


        # --- Data Cleaning and Type Conversion for Standard Columns ---
        # Now work with the standard column names defined in expected_data_header_simple_v2 (or what was read)

        # Convert 'Fecha' to datetime
        if 'Fecha' in df.columns:
             df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce', dayfirst=True) # Assuming day first (dd-mm-yyyy, dd/mm/yyyy)
        else:
             st.warning("Fuel parse (Simple V2): Columna estándar 'Fecha' no encontrada. Las filas no tienen fecha.")
             df['Fecha'] = pd.NaT # Add Date column as NaT if missing

        # Ensure string columns exist and are cleaned
        standard_string_cols = ['Equipo', 'CodigoCombustible', 'HsKm', 'Comentarios']
        for col in standard_string_cols:
            if col in df.columns:
                 df[col] = df[col].astype(str).str.strip().fillna('')
            else:
                 st.warning(f"Fuel parse (Simple V2): Columna estándar '{col}' no encontrada. Usando cadena vacía.")
                 df[col] = ''


        # Convert numeric columns using safe_float_parse
        standard_numeric_cols_to_parse = ['LtsIngreso', 'LtsEgreso', 'CostoUnitarioLt']
        for col in standard_numeric_cols_to_parse:
             if col in df.columns:
                  df[col] = df[col].apply(safe_float_parse)
             else:
                  st.warning(f"Fuel parse (Simple V2): Columna numérica estándar '{col}' no encontrada. Usando 0.0.")
                  df[col] = 0.0

        # Convert 'HsKm' specifically to numeric for ratio calculations
        # Use the potentially cleaned 'HsKm' string column as input for parsing
        if 'HsKm' in df.columns:
             df['Hs/Km_Numeric'] = df['HsKm'].apply(safe_float_parse)
        else: # HsKm was not in original columns and was added as empty string, parse from that default
             df['Hs/Km_Numeric'] = df['HsKm'].apply(safe_float_parse) # This will be 0.0
        # Ensure it's truly numeric; errors become NaN, then fill NaNs with 0.0 for calculations
        df['Hs/Km_Numeric'] = pd.to_numeric(df['Hs/Km_Numeric'], errors='coerce').fillna(0.0)


        # Calculate Costo Total Egreso (ensure columns are numeric)
        if 'LtsEgreso' in df.columns and 'CostoUnitarioLt' in df.columns:
             df['Costo Total Egreso'] = df['LtsEgreso'].fillna(0.0) * df['CostoUnitarioLt'].fillna(0.0) # Use standard names
        else:
             st.warning("Fuel parse (Simple V2): Columnas 'LtsEgreso' o 'CostoUnitarioLt' faltantes. No se puede calcular 'Costo Total Egreso'.")
             df['Costo Total Egreso'] = 0.0


        # Map 'CodigoCombustible' to 'Tipo de Comb.' using the mapping found in SETUP
        if 'CodigoCombustible' in df.columns:
             # Ensure 'CodigoCombustible' is string and stripped before mapping
             df['CodigoCombustible'] = df['CodigoCombustible'].astype(str).str.strip()
             df['Tipo de Comb.'] = df['CodigoCombustible'].map(fuel_type_mapping).fillna('Desconocido')
        else:
             st.warning("Fuel parse (Simple V2): Columna estándar 'CodigoCombustible' no encontrada. 'Tipo de Comb.' será 'Desconocido'.")
             df['Tipo de Comb.'] = 'Desconocido'


        # --- Filter out rows where core identifiers are empty/invalid after processing ---
        # Now that cleaning and type conversions are done, filter invalid rows.
        initial_rows = len(df)
        df_filtered = df.copy()

        # Apply filters only if the columns exist
        if 'Fecha' in df_filtered.columns:
             df_filtered = df_filtered[df_filtered['Fecha'].notna()].copy() # Keep rows with valid date
        else: # If Fecha column was never added, add it as NaT for consistent filtering logic below
             df_filtered['Fecha'] = pd.NaT
             df_filtered = df_filtered[df_filtered['Fecha'].notna()].copy() # This will remove all rows


        if 'Equipo' in df_filtered.columns:
             df_filtered = df_filtered[df_filtered['Equipo'].astype(str).str.strip() != ''].copy() # Keep rows with non-empty Equipo
        else: # If Equipo column was never added, add it as empty string for consistent filtering
             df_filtered['Equipo'] = ''
             df_filtered = df_filtered[df_filtered['Equipo'].astype(str).str.strip() != ''].copy() # This will remove all rows


        if 'CodigoCombustible' in df_filtered.columns:
             df_filtered = df_filtered[df_filtered['CodigoCombustible'].astype(str).str.strip() != ''].copy() # Keep rows with non-empty CodigoCombustible
        else: # If CodigoCombustible was never added, add it as empty string
             df_filtered['CodigoCombustible'] = ''
             df_filtered = df_filtered[df_filtered['CodigoCombustible'].astype(str).str.strip() != ''].copy() # This will remove all rows


        if len(df_filtered) < initial_rows:
            st.warning(f"Fuel parse (Simple V2): {initial_rows - len(df_filtered)} filas fueron omitidas por falta de 'Fecha', 'Equipo', o 'CodigoCombustible' válidos después de la limpieza.")

        df = df_filtered # Use the filtered DataFrame going forward


        # --- RENAME columns to match internal application standard names ---
        # Use the standard names (from expected_data_header_simple_v2) that are now DF columns.
        rename_map_to_internal = {
            'Equipo': 'Equipo/Int.', # Renamed
            'CodigoCombustible': 'Codigo', # Renamed, internal standard for item code
            'LtsIngreso': 'Lts Ingreso', # Renamed
            'LtsEgreso': 'Lts Egreso',   # Renamed
            'CostoUnitarioLt': 'Costo Unitario Lt', # Renamed
            # HsKm, Comentarios, Tipo de Comb., Costo Total Egreso, Hs/Km_Numeric match internal names.
        }

        # Apply renaming only if the source column exists in the current dataframe columns
        cols_to_rename_actual = {src: dst for src, dst in rename_map_to_internal.items() if src in df.columns}
        if cols_to_rename_actual:
             df.rename(columns=cols_to_rename_actual, inplace=True)


        # --- Ensure ALL required internal standard columns exist and are in a defined order ---
        # Define the final desired order of columns.
        internal_standard_cols_ordered = [
            'Fecha', 'Equipo/Int.', 'Codigo', 'Lts Ingreso', 'Lts Egreso', 'HsKm',
            'Hs/Km_Numeric', 'Tipo de Comb.', 'Comentarios', 'Costo Unitario Lt', 'Costo Total Egreso'
        ]

        # Add any missing standard columns with default values to ensure structure
        for col in internal_standard_cols_ordered:
            if col not in df.columns:
                 if col in ['Lts Ingreso', 'Lts Egreso', 'Hs/Km_Numeric', 'Costo Unitario Lt', 'Costo Total Egreso']:
                     df[col] = 0.0 # Numeric defaults
                 elif col == 'Fecha':
                     df[col] = pd.NaT # Datetime default
                 else: # 'Equipo/Int.', 'Codigo', 'HsKm', 'Tipo de Comb.', 'Comentarios' defaults
                    df[col] = ''

        # Reorder the DataFrame to the standard internal order, only keeping columns that actually exist.
        # This ensures the returned DF has the expected columns even if empty.
        final_df_ordered = df.loc[:, [col for col in internal_standard_cols_ordered if col in df.columns]]

        # Ensure the initial stock dictionary is returned, using the values found in SETUP
        initial_stock_result = {'GASOIL': saldo_gasoil, 'NAFTA': saldo_nafta}


        return final_df_ordered, initial_stock_result

    except Exception as e:
        st.error(f"Error general o de procesamiento en archivo de Combustible (Simple V2): {e}. Verifica si sigue la estructura simple con líneas SETUP y cabecera de datos ('Fecha,Equipo,...').")
        # st.error(traceback.format_exc()) # Uncomment for detailed debugging
        # On general parse failure, return None as originally intended for critical errors
        return None # Return None on critical parse failure


# --- Initialize Session State ---
# Use clear variable names
# Initialize with None; parsing will set to DataFrame/dict or keep None on critical failure
if 'stock_df' not in st.session_state: st.session_state.stock_df = None
if 'eco_data' not in st.session_state: st.session_state.eco_data = None # Will be a dictionary
if 'fuel_df' not in st.session_state: st.session_state.fuel_df = None
if 'fuel_initial_stock' not in st.session_state: st.session_state.fuel_initial_stock = None # Will be a dictionary

# Add unique file identifiers to session state to track if a *new* file has been uploaded
if 'last_stock_file_id' not in st.session_state: st.session_state.last_stock_file_id = None
if 'last_eco_file_id' not in st.session_state: st.session_state.last_eco_file_id = None
if 'last_fuel_file_id' not in st.session_state: st.session_state.last_fuel_file_id = None

# Helper to get a unique ID for an uploaded file
def get_file_id(file):
     if file is None: return None
     try:
         # Using only name and size as hash can be problematic with Streamlit's file handling
         return f"{file.name}-{file.size}"
     except Exception:
         return None # Return None if file object is unexpectedly structured


# --- File Upload and Parsing Logic ---
# Get current IDs from file uploader widgets
current_stock_file = stock_file # Use variable names from uploaders
current_eco_file = eco_file
current_fuel_file = fuel_file

current_stock_file_id = get_file_id(current_stock_file)
current_eco_file_id = get_file_id(current_eco_file)
current_fuel_file_id = get_file_id(current_fuel_file)

# Parse Stock File if a NEW file is uploaded (ID changed)
if current_stock_file_id != st.session_state.last_stock_file_id:
    st.session_state.stock_df = None # Clear old data *before* trying to parse new
    st.session_state.last_stock_file_id = current_stock_file_id # Update stored ID

    if current_stock_file is not None:
        # parse_stock_visma returns DF (structured empty or populated) or None on critical error
        st.session_state.stock_df = parse_stock_visma(current_stock_file)
        if st.session_state.stock_df is not None: # Check if parsing was NOT a critical error
            if not st.session_state.stock_df.empty:
                 st.sidebar.success(f"Archivo de Stock cargado y procesado ({len(st.session_state.stock_df)} items válidos).")
            else:
                 st.sidebar.warning("Archivo de Stock cargado, pero no se encontraron items válidos o no cero.")
        # If parse_stock_visma returned None (critical error), session_state.stock_df remains None and an error is shown by the parser


# Parse ECO File if a NEW file is uploaded (ID changed)
if current_eco_file_id != st.session_state.last_eco_file_id:
    st.session_state.eco_data = None # Clear old data
    st.session_state.last_eco_file_id = current_eco_file_id # Update stored ID

    if current_eco_file is not None:
        # parse_eco_visma returns dict (empty structured or populated) or None on critical error
        st.session_state.eco_data = parse_eco_visma(current_eco_file)
        if st.session_state.eco_data is not None: # Check if parsing was NOT a critical error
             # Check if any non-zero values were parsed (now checking the actual parsed data, which might be the zeroed structure)
             expected_keys_eco = ['Movilizacion', 'Costos Directos', 'Costos Indirectos/Generales', 'Utilidades'] # These must match keys in eco_data structure
             has_eco_values = (
                 any(abs(st.session_state.eco_data.get('invierno',{}).get(k, 0.0)) > 1e-9 for k in expected_keys_eco) or
                 any(abs(st.session_state.eco_data.get('verano',{}).get(k, 0.0)) > 1e-9 for k in expected_keys_eco)
             )

             if has_eco_values:
                st.sidebar.success("Archivo de Presupuesto cargado y procesado.")
             else:
                # If eco_data is not None but has no non-zero values, it's likely the zeroed structure from parser
                st.sidebar.warning("Archivo de Presupuesto procesado, pero no se encontraron costos no cero para las categorías esperadas ('Movilizacion', 'CostosDirectos', ...).")
        # If parse_eco_visma returned None, session_state.eco_data remains None and an error is shown by the parser


# Parse Fuel File if a NEW file is uploaded (ID changed)
if current_fuel_file_id != st.session_state.last_fuel_file_id:
    st.session_state.fuel_df = None # Clear old data
    st.session_state.fuel_initial_stock = None # Clear old data
    st.session_state.last_fuel_file_id = current_fuel_file_id # Update stored ID

    if current_fuel_file is not None:
        # parse_fuel_log returns a tuple (DF, dict) or None on critical error
        fuel_parse_result = parse_fuel_log(current_fuel_file)

        if fuel_parse_result is not None: # Check if parsing itself was NOT a critical error
             st.session_state.fuel_df, st.session_state.fuel_initial_stock = fuel_parse_result

             # Check if parsed DF is usable OR if initial stock has non-zero values
             is_df_usable = st.session_state.fuel_df is not None and not st.session_state.fuel_df.empty # fuel_df will be structured empty DF if no data rows
             # Check if initial stock dict is valid and has at least one non-zero value
             has_initial_stock_values = st.session_state.fuel_initial_stock is not None and (
                  abs(st.session_state.fuel_initial_stock.get('GASOIL', 0.0)) > 1e-9 or
                  abs(st.session_state.fuel_initial_stock.get('NAFTA', 0.0)) > 1e-9
             )

             if is_df_usable:
                  st.sidebar.success(f"Archivo de Combustible cargado y procesado ({len(st.session_state.fuel_df)} registros de datos válidos).")
             elif has_initial_stock_values:
                  # No valid data rows, but initial stock was read. fuel_df will be an empty structured DF.
                  st.sidebar.warning("Archivo de Combustible procesado. Saldo inicial leído, pero no se encontraron registros de movimientos de datos válidos.")
             else:
                  # No valid data rows and zero initial stock. fuel_df will be empty, stock dict will be zeroed.
                  st.sidebar.warning("Archivo de Combustible procesado, pero no contiene datos de movimientos válidos ni saldo inicial no cero.")
        # If parse_fuel_log returned None (critical error), session_state.fuel_df and fuel_initial_stock remain None and an error is shown by the parser


# --- Retrieve data from Session State ---
# Explicitly check for None after retrieval and provide default structures
stock_df = st.session_state.get('stock_df')
if stock_df is None:
    # Define the full expected columns for an empty stock DF if parsing failed critically
    empty_stock_cols = ['Codigo', 'Producto', 'Categoria', 'CantidadActual', 'CostoUnitario', 'Valor Total Item', 'Ubicacion', 'STOCK_MINIMO']
    stock_df = pd.DataFrame(columns=empty_stock_cols) # Set a default empty structured DF

eco_data = st.session_state.get('eco_data') # Retrieve value from state
if eco_data is None:
    # Define the full expected structure for empty eco_data if parsing failed critically
    expected_keys_eco = ['Movilizacion', 'Costos Directos', 'Costos Indirectos/Generales', 'Utilidades']
    eco_data = {
       'invierno': {key: 0.0 for key in expected_keys_eco},
       'verano': {key: 0.0 for key in expected_keys_eco}
    } # Set a default empty structured dict

fuel_df = st.session_state.get('fuel_df') # Retrieve value from state
if fuel_df is None:
     # Define the full expected columns for an empty fuel DF if parsing failed critically
     empty_fuel_cols = ['Fecha', 'Equipo/Int.', 'Codigo', 'Lts Ingreso', 'Lts Egreso', 'HsKm', 'Comentarios', 'Tipo de Comb.', 'Hs/Km_Numeric', 'Costo Unitario Lt', 'Costo Total Egreso']
     fuel_df = pd.DataFrame(columns=empty_fuel_cols) # Set a default empty structured DF

fuel_initial_stock = st.session_state.get('fuel_initial_stock') # Retrieve value from state
if fuel_initial_stock is None:
    fuel_initial_stock = {'GASOIL': 0.0, 'NAFTA': 0.0} # Set a default zeroed dict

# Now the variables stock_df, eco_data, fuel_df, fuel_initial_stock are guaranteed not to be None
# They will be either the parsed data (potentially empty but structured) or the default empty structures


# --- Display Results in Tabs ---
# Use the session state variables directly which are now guaranteed to be DataFrame/dict (possibly empty but structured) or None only on critical parse failure
tab_costs, tab_ratios, tab_waterfall, tab_stock, tab_budget = st.tabs([
    "Costos Operativos",
    "Ratios Consumo / Eficiencia",
    "Gráfico de Cascada",
    "Gestión de Stock",
    "Creación de Presupuestos"
])


# --- TAB: Costos Operativos ---
with tab_costs:
    st.header("Análisis de Costos Operativos")

    # Get cost data, using .get({}, {}) for safety if eco_data is missing 'invierno'/'verano' keys (less likely with default structure)
    costs_inv = eco_data.get('invierno', {})
    costs_verano = eco_data.get('verano', {})

    # Define the order of categories for display and calculation totals
    ordered_categories_internal_keys = ['Movilizacion', 'Costos Directos', 'Costos Indirectos/Generales', 'Utilidades']
    # Check if any relevant non-zero data exists in the costs structures
    # Check if costs_inv/verano are dicts before checking keys
    has_inv_data_to_display = isinstance(costs_inv, dict) and any(abs(costs_inv.get(k, 0.0)) > 1e-9 for k in ordered_categories_internal_keys)
    has_verano_data_to_display = isinstance(costs_verano, dict) and any(abs(costs_verano.get(k, 0.0)) > 1e-9 for k in ordered_categories_internal_keys)


    if has_inv_data_to_display or has_verano_data_to_display:
        st.subheader("Costos por Categoría (Basado en Archivo de Presupuesto Simple)")
        st.info("La clasificación en Movilización, Directos, Indirectos, Utilidades se basa directamente en las filas leídas del archivo de presupuesto simple cargado.")

        col1, col2 = st.columns(2)

        if has_inv_data_to_display:
            with col1:
                 st.subheader("Temporada Invierno")
                 total_inv = 0.0 # Initialize total
                 # Iterate over ordered keys, ensuring they exist in costs_inv dict
                 for key in [k for k in ordered_categories_internal_keys if k in costs_inv]:
                      value = costs_inv.get(key, 0.0)
                      st.write(f"- {key}: ${value:,.2f}") # Use key from the ordered list
                      total_inv += value # Sum value for the total
                 st.subheader(f"Total Calculado Invierno: ${total_inv:,.2f}")

        if has_verano_data_to_display:
            # Display verano costs in col2 if invierno data is present, otherwise in col1
            with col2 if has_inv_data_to_display else col1:
                st.subheader("Temporada Verano")
                total_verano = 0.0 # Initialize total
                # Iterate over ordered keys, ensuring they exist in costs_verano dict
                for key in [k for k in ordered_categories_internal_keys if k in costs_verano]:
                    value = costs_verano.get(key, 0.0)
                    st.write(f"- {key}: ${value:,.2f}") # Use key from the ordered list
                    total_verano += value # Sum value for the total
                st.subheader(f"Total Calculado Verano: ${total_verano:,.2f}")

        # Specific message if file was processed but no non-zero relevant data found
        # Check if eco_data is a dictionary and it was likely populated (even with zeros)
        elif isinstance(eco_data, dict) and (eco_data.get('invierno') is not None or eco_data.get('verano') is not None): # Check if keys exist even if values are zeroed
             st.info("El archivo de presupuesto simple fue procesado, pero no contiene datos de costos no cero para las categorías esperadas.")

    else:
        # Message when no file was uploaded or parsing failed critically
        st.info("Por favor, carga el archivo de `presupuesto_simple.csv` para ver el análisis de costos.")

    st.markdown("---")
    st.subheader("Sobre la Definición de Costos Operativos")
    st.info("El análisis de costos directos variables operativos, fijos directos/indirectos, etc. depende de cómo se clasifiquen y calculen estos costos en el archivo de entrada. El archivo simple clasifica por tipo principal (Directos, Indirectos, etc.), pero la determinación de su variabilidad, o fijeza debe hacerse al generar esos totales en el archivo origen (ej. Excel).")


# --- TAB: Ratios Consumo / Eficiencia ---
with tab_ratios:
    st.header("Ratios de Consumo y Eficiencia")

    # Check if fuel_df is loaded and has the required columns
    required_fuel_cols_ratios = ['Equipo/Int.', 'Tipo de Comb.', 'Lts Egreso', 'Hs/Km_Numeric', 'Comentarios', 'Costo Unitario Lt', 'Costo Total Egreso', 'Fecha']

    # Check if fuel_df is a DataFrame and has the necessary columns before proceeding
    if isinstance(fuel_df, pd.DataFrame) and all(col in fuel_df.columns for col in required_fuel_cols_ratios):

         if not fuel_df.empty: # Check if the DF actually contains rows

            st.info("Los datos de consumo y eficiencia se basan en los registros con `LtsEgreso > 0.0` en el archivo de combustible.")

            # --- Consumo Total por Equipo (Tabla y Gráfico de Barras) ---
            st.subheader("Consumo Total de Combustible por Equipo (Lts)")

            # Filter consumption rows: Lts Egreso > 0.0 and NOT marked as PRESTAMO
            # Ensure Comentarios is string before contains check
            consumption_mask = (fuel_df['Lts Egreso'] > 1e-9) & (~fuel_df['Comentarios'].astype(str).str.contains('PRESTAMO', na=False, case=False))
            consumption_df = fuel_df[consumption_mask].copy()

            if not consumption_df.empty:
                 # Ensure required columns exist in the filtered dataframe (should be, but belt and suspenders)
                 if all(col in consumption_df.columns for col in ['Equipo/Int.', 'Lts Egreso', 'Tipo de Comb.']):
                      fuel_consumption_per_equipment = consumption_df.groupby('Equipo/Int.').agg(
                          Total_Liters=('Lts Egreso', 'sum'),
                          # Use .mode() with [0] to get the most frequent fuel type for this equipment, default if empty
                          Tipo=('Tipo de Comb.', lambda x: x.mode()[0] if not x.empty else 'Multiple/Unknown')
                      ).reset_index()

                      if not fuel_consumption_per_equipment.empty and fuel_consumption_per_equipment['Total_Liters'].sum() > 1e-9: # Check if there's any consumption
                          st.dataframe(fuel_consumption_per_equipment[['Equipo/Int.', 'Tipo', 'Total_Liters']].sort_values(by='Total_Liters', ascending=False).reset_index(drop=True), use_container_width=True) # Added drop=True to avoid index

                          st.markdown("#### Gráfico de Consumo por Equipo")
                          fig_consumo_equipo = px.bar(
                              fuel_consumption_per_equipment.sort_values(by='Total_Liters', ascending=False),
                              x='Equipo/Int.',
                              y='Total_Liters',
                              color='Tipo', # Color bars by fuel type
                              title='Consumo Total de Litros por Equipo',
                              labels={'Equipo/Int.': 'Equipo / Interno', 'Total_Liters': 'Litros Consumidos'}
                          )
                          fig_consumo_equipo.update_layout(xaxis={'categoryorder':'total descending'}) # Ensure bars are sorted
                          st.plotly_chart(fig_consumo_equipo, use_container_width=True)

                      else:
                           st.info("El consumo total calculado por equipo es cero o insignificante.")
                 else:
                      st.warning("Columnas estándar requeridas ('Equipo/Int.', 'Lts Egreso' o 'Tipo de Comb.') faltantes en los datos filtrados de consumo.")
            else:
                 st.info("No se registraron egresos de combustible válidos (Lts Egreso > 0.0 y no es préstamo) para equipos en el archivo cargado.")

            st.markdown("---")

            # --- Costo de Combustible por Equipo (Bar Chart) ---
            st.subheader("Costo de Combustible por Equipo ($)")
            st.warning("Este cálculo requiere que la columna `CostoUnitarioLt` contenga valores numéricos válidos, > $0, en las filas con egresos (`LtsEgreso > 0`). Asegúrate que las líneas de egreso (`LtsEgreso > 0`) se correspondan a un costo.")

            # Calculate total cost per equipment from rows with positive egress and positive Costo Total Egreso
            cost_consumption_mask = (fuel_df['Costo Total Egreso'] > 1e-9) & (~fuel_df['Comentarios'].astype(str).str.contains('PRESTAMO', na=False, case=False))
            cost_consumption_df = fuel_df[cost_consumption_mask].copy()

            if not cost_consumption_df.empty:
                 # Ensure required columns exist in the filtered dataframe
                 if all(col in cost_consumption_df.columns for col in ['Equipo/Int.', 'Costo Total Egreso', 'Tipo de Comb.']):

                      fuel_cost_per_equipment = cost_consumption_df.groupby('Equipo/Int.').agg(
                          Total_Cost=('Costo Total Egreso', 'sum'),
                          # Use .mode() with [0] to get the most frequent fuel type for this equipment
                          Tipo=('Tipo de Comb.', lambda x: x.mode()[0] if not x.empty else 'Multiple/Unknown')
                      ).reset_index()

                      if not fuel_cost_per_equipment.empty and fuel_cost_per_equipment['Total_Cost'].sum() > 1e-9: # Check if there's any cost
                          # Display Bar Chart for Cost by Equipment
                          st.markdown("#### Gráfico de Costo por Equipo")
                          fig_cost_per_equipo = px.bar(
                              fuel_cost_per_equipment.sort_values(by='Total_Cost', ascending=False),
                              x='Equipo/Int.',
                              y='Total_Cost',
                              color='Tipo', # Color bars by fuel type
                              title='Costo Total de Combustible por Equipo ($)',
                              labels={'Equipo/Int.': 'Equipo / Interno', 'Total_Cost': 'Costo Total ($)'}
                          )
                          fig_cost_per_equipo.update_layout(xaxis={'categoryorder':'total descending'}, yaxis_title="Costo ($)")
                          st.plotly_chart(fig_cost_per_equipo, use_container_width=True)
                      else:
                           st.info("El costo total calculado por equipo es cero o insignificante (verifica valores de `CostoUnitarioLt` y `LtsEgreso`).")
                 else:
                     st.warning("Columnas requeridas ('Equipo/Int.', 'Costo Total Egreso' o 'Tipo de Comb.') faltantes en los datos filtrados de costo.")
            else:
                 st.info("No se registraron egresos de combustible con costo válido (> $0) por equipo.")


            st.markdown("---")

            # --- Ratios de Eficiencia (Lts/Hs o Lts/Km) Table ---
            st.subheader("Registros de Consumo con Uso (Hs/Km) registrado")
            st.warning("Para ver ratios, la columna `HsKm` debe contener un **valor numérico positivo** (> 0.0) asociado a los egresos de combustible.")

            # Filter for rows where Lts Egreso > 0.0 AND Hs/Km_Numeric is valid (> 0)
            usage_recorded_mask = (fuel_df['Lts Egreso'] > 1e-9) & (fuel_df['Hs/Km_Numeric'].notna()) & (fuel_df['Hs/Km_Numeric'] > 1e-9)
            usage_recorded_df = fuel_df[usage_recorded_mask].copy()

            if not usage_recorded_df.empty:
                 # Ensure required columns exist in the filtered dataframe before calculating ratio
                 required_cols_for_ratio_calc = ['Lts Egreso', 'Hs/Km_Numeric']
                 if all(col in usage_recorded_df.columns for col in required_cols_for_ratio_calc):
                      # Calculate ratio - Hs/Km_Numeric is guaranteed > 0 and notna() here by the mask
                      usage_recorded_df['Lts / Uso'] = usage_recorded_df['Lts Egreso'] / usage_recorded_df['Hs/Km_Numeric']

                      st.markdown("#### Registros con Ratios Calculados")
                      display_cols = ['Fecha', 'Equipo/Int.', 'Lts Egreso', 'HsKm', 'Lts / Uso', 'Tipo de Comb.', 'Costo Total Egreso', 'Comentarios'] # Added Costo Total Egreso, ensure Equipo/Int. and Tipo de Comb exist
                      display_cols = [col for col in display_cols if col in usage_recorded_df.columns] # Filter cols by actual presence in DF

                      if display_cols:
                           # Display DataFrame with calculated ratio
                           st.dataframe(usage_recorded_df[display_cols].sort_values(by='Fecha', ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
                           st.info("Estos son los registros donde se pudo calcular un ratio (Lts / Uso). Si 'HsKm' representa horas, el ratio es Lts/Hora. Si representa kilómetros, es Lts/Km.")
                      else:
                           st.warning("Columnas necesarias para mostrar registros con uso válido faltantes ('Lts Egreso', 'HsKm', 'Hs/Km_Numeric').")

                      # --- Efficiency Trend Line Chart ---
                      st.markdown("#### Tendencia de Eficiencia (Lts / Uso) por Equipo")
                      st.info("Mostrando tendencia solo para equipos con **2 o más registros de ratio válidos** para que se pueda dibujar una línea.")

                      # Identify equipment with multiple data points for a line chart to make sense
                      equipment_usage_counts = usage_recorded_df['Equipo/Int.'].value_counts() if 'Equipo/Int.' in usage_recorded_df.columns else pd.Series()
                      equipos_for_trend = equipment_usage_counts[equipment_usage_counts >= 2].index.tolist()

                      if equipos_for_trend and 'Fecha' in usage_recorded_df.columns and 'Lts / Uso' in usage_recorded_df.columns:
                           # Allow selecting which equipment to show
                           selected_equipos = st.multiselect(
                               "Selecciona equipos para la tendencia:",
                               options=equipos_for_trend,
                               default=equipos_for_trend[:min(len(equipos_for_trend), 3)] # Select first 3 by default if available
                           )

                           if selected_equipos:
                               # Filter the data for selected equipment and sort by date
                               trend_df = usage_recorded_df[usage_recorded_df['Equipo/Int.'].isin(selected_equipos)].sort_values(by='Fecha').copy()

                               if not trend_df.empty:
                                    fig_trend_eficiencia = px.line(
                                        trend_df,
                                        x='Fecha',
                                        y='Lts / Uso',
                                        color='Equipo/Int.',
                                        title='Tendencia de Eficiencia (Lts / Uso) por Equipo',
                                        labels={'Lts / Uso': 'Litros por Unidad de Uso'}, # Generic label
                                        markers=True # Show data points
                                    )
                                    fig_trend_eficiencia.update_layout(xaxis_title="Fecha", yaxis_title="Lts / Uso")
                                    st.plotly_chart(fig_trend_eficiencia, use_container_width=True)
                               else:
                                   st.info("No hay datos de ratio válidos para los equipos seleccionados.")
                           else:
                               st.info("Selecciona al menos un equipo para ver la tendencia de eficiencia.")

                      else:
                            st.info("No hay equipos con suficientes registros (2 o más) con ratios válidos y fecha para mostrar una tendencia, o faltan columnas ('Fecha', 'Lts / Uso', 'Equipo/Int.').")

                 else:
                      st.warning("Columnas necesarias para calcular ratios ('Lts Egreso', 'Hs/Km_Numeric') faltantes.")

            else:
                 st.info("No hay registros de egresos de combustible válidos (> 0.0) que también tengan un valor numérico válido y mayor a cero en la columna 'HsKm'.")


         else: # fuel_df exists but is empty
             st.info("El archivo de Combustible fue cargado, pero no contiene ninguna fila de datos válida.")
             st.warning("Asegúrate de que el archivo `combustible_simple_v2.csv` tenga filas debajo de la cabecera de datos ('Fecha,Equipo,...') que no estén vacías o mal formadas, y que incluyan al menos una `Fecha`, un `Equipo` y un `CodigoCombustible` válidos.")

    else: # fuel_df is None or not a DataFrame, or missing required columns after parsing (handled by the initial check)
        # Check specifically if it's a DataFrame that's missing columns vs. None
        if isinstance(fuel_df, pd.DataFrame) and required_fuel_cols_ratios and not all(col in fuel_df.columns for col in required_fuel_cols_ratios):
            missing_cols = [col for col in required_fuel_cols_ratios if col not in fuel_df.columns]
            st.error(f"El archivo de Combustible fue cargado, pero faltan columnas esenciales para el análisis de ratios: {', '.join(missing_cols)}. Revisa el formato.")
        else: # It's None, or some other state indicating critical failure/no file
            st.info("Por favor, carga el archivo `combustible_simple_v2.csv` para ver los datos de consumo y ratios.")


    st.markdown("---")
    st.subheader("Sugerencias para Mejorar Ratios de Consumo y Eficiencia")
    st.markdown("""
        - Asegúrate de que para CADA registro de Egreso (`LtsEgreso > 0.0`) se complete un **valor numérico positivo** (`> 0`) en la columna **`HsKm`** para calcular ratios Lts/Uso.
        - La columna **`CostoUnitarioLt`** es necesaria para los gráficos de costo. Ingresa el costo en la línea de ingreso (`LtsIngreso > 0`) del combustible.
        - La funcionalidad actual calcula Costo Total Egreso por línea (`LtsEgreso * CostoUnitarioLt`). Se asume que `CostoUnitarioLt` en una fila de egreso representa el costo del combustible egresado en ese movimiento. Para un cálculo de costo de egreso basado en el costo promedio del stock en el momento del egreso, se necesitaría un registro de stock histórico más complejo.
    """)


# --- TAB: Gráfico de Cascada ---
with tab_waterfall:
    st.header("Visualización de Costos (Gráfico de Cascada)")

    # Get cost data from eco_data, defaulting to empty dicts if keys are missing (handled by default structure now)
    costs_inv = eco_data.get('invierno', {})
    costs_verano = eco_data.get('verano', {})

    # Use the defined order of categories for waterfall steps
    ordered_categories_internal_keys = ['Movilizacion', 'Costos Directos', 'Costos Indirectos/Generales', 'Utilidades']

    # Check if eco_data is a dictionary and if there is any non-zero relevant data for Invierno breakdown
    has_inv_data_to_breakdown = isinstance(costs_inv, dict) and any(abs(costs_inv.get(k, 0.0)) > 1e-9 for k in ordered_categories_internal_keys)

    if has_inv_data_to_breakdown:
        st.subheader("Desglose de Costos Temporada Invierno")

        # Data for the Waterfall chart for Invierno Breakdown
        waterfall_labels_inv = []
        waterfall_values_inv = []
        waterfall_measures_inv = []
        waterfall_text_inv = []

        # Add starting base
        waterfall_labels_inv.append("Inicio Base")
        waterfall_values_inv.append(0.0) # Starts at 0
        waterfall_measures_inv.append('absolute')
        waterfall_text_inv.append("$0")


        relative_steps_added = 0 # Count how many relative steps (categories) are non-zero
        # Iterate over ordered keys, ensuring they exist in costs_inv dict
        for internal_key in [k for k in ordered_categories_internal_keys if k in costs_inv]:
             label = internal_key
             value = costs_inv.get(internal_key, 0.0) # Get value, default to 0.0

             if abs(value) > 1e-9: # Only add steps with non-zero values
                 waterfall_labels_inv.append(label)
                 waterfall_values_inv.append(value)
                 waterfall_measures_inv.append('relative') # Relative step
                 waterfall_text_inv.append(f"${value:,.0f}") # Text for display on chart
                 relative_steps_added += 1 # Increment counter


        # Calculate the total *after* processing all categories (summing only keys present in costs_inv)
        total_inv_calculated = sum(costs_inv.get(k, 0.0) for k in [k for k in ordered_categories_internal_keys if k in costs_inv])


        # Add the final total bar ONLY if there were relative steps OR the total is non-zero
        if relative_steps_added > 0 or abs(total_inv_calculated) > 1e-9: # Ensure there is something meaningful to show
             # Add the final 'Total Invierno' bar
             waterfall_labels_inv.append("Total Invierno Calculado")
             waterfall_values_inv.append(total_inv_calculated) # Final value
             waterfall_measures_inv.append('total') # This MUST be 'total' measure for Plotly waterfall end bar
             waterfall_text_inv.append(f"${total_inv_calculated:,.0f}")


        # Check if there are enough points to plot a waterfall (at least base + total, or base + 1 relative)
        # len(waterfall_labels_inv) > 1 is a sufficient check after adding base and potential relative/total bars
        if len(waterfall_labels_inv) > 1:
             fig_inv = go.Figure(go.Waterfall(
                 name = "Costos Invierno",
                 orientation = "v", # Vertical waterfall
                 measure = waterfall_measures_inv,
                 x = waterfall_labels_inv, # Categories / Step labels
                 y = waterfall_values_inv, # Values for each step
                 textposition = "outside", # Position of text labels on bars
                 text = waterfall_text_inv, # Text content for labels
                 connector = {"line": {"color": "rgb(63, 63, 63)"}}, # Style connector lines
                 # Marker colors: Default for relative, specify for 'total' bars via 'totals'
                 # increasing = {"marker":{"color": "forestgreen"}}, # Optional: custom color for increasing steps
                 # decreasing = {"marker":{"color": "crimson"}}, # Optional: custom color for decreasing steps
                 totals = {"marker": {"color": "royalblue", "line": {"color": "royalblue", "width": 2}}} # Style for 'total' bars (Start/End totals)
             ))

             fig_inv.update_layout(
                 title = "Gráfico de Cascada de Costos - Temporada Invierno (Desglose por Categoría)",
                 showlegend = False,
                 yaxis_title="Monto ($)",
                 margin=dict(l=20, r=20, t=40, b=20), # Adjust margins
                 hovermode="x unified", # Improve hover experience
             )
             st.plotly_chart(fig_inv, use_container_width=True)

        # Message if data is present but totals/steps are all zero/insignificant
        # Check if costs_inv is a dict before checking its content
        elif isinstance(costs_inv, dict) and any(abs(costs_inv.get(k, 0.0)) > 1e-9 for k in [k for k in ordered_categories_internal_keys if k in costs_inv]):
             st.info(f"El archivo de presupuesto para Invierno fue procesado. Sin embargo, las categorías individuales suman cero o son muy pequeñas para visualizar un desglose detallado en el gráfico de cascada.")
             st.write(f"Total calculado Invierno: ${total_inv_calculated:,.2f}")


    st.markdown("---")
    # --- Comparison Waterfall (Invierno vs. Verano) ---
    st.subheader("Comparación de Costos: Invierno vs. Verano")

    # Check if data for *both* seasons is available and relevant (ensure costs_inv/verano are dicts)
    has_both_seasons_data_parsed = isinstance(costs_inv, dict) and isinstance(costs_verano, dict)

    # Calculate total sums for comparison (summing only keys present in respective dicts)
    total_invierno_comp = sum(costs_inv.get(k, 0.0) for k in [k for k in ordered_categories_internal_keys if k in costs_inv])
    total_verano_comp = sum(costs_verano.get(k, 0.0) for k in [k for k in ordered_categories_internal_keys if k in costs_verano])


    # Check if the overall totals are significantly different OR if category differences exist
    difference_between_totals_exists = abs(total_invierno_comp - total_verano_comp) > 1e-9

    category_differences_exist_in_comparison = False
    if has_both_seasons_data_parsed:
         for category in ordered_categories_internal_keys:
              # Check if the category key exists in *both* dictionaries before comparing
              if category in costs_inv and category in costs_verano:
                   inv_val = costs_inv.get(category, 0.0)
                   verano_val = costs_verano.get(category, 0.0)
                   if abs(verano_val - inv_val) > 1e-9: # Check for significant difference
                        category_differences_exist_in_comparison = True
                        break # Found at least one significant difference

    # Trigger comparison waterfall display if both seasons data dictionaries are available AND there's something interesting to show
    # Interesting could be: the total changes, or individual categories change even if total doesn't, or just show totals if both exist.
    if has_both_seasons_data_parsed and (difference_between_totals_exists or category_differences_exist_in_comparison or abs(total_invierno_comp) > 1e-9 or abs(total_verano_comp) > 1e-9):

         # Data for the Comparison Waterfall
         diff_labels = []
         diff_values = []
         diff_measures = []
         diff_text_values = []

         # Start with the base Winter total IF it's non-zero, or if the comparison makes sense
         # Only add the base bar if the starting total is non-zero OR there's a total change OR category changes will be shown.
         if abs(total_invierno_comp) > 1e-9 or difference_between_totals_exists or category_differences_exist_in_comparison:
              diff_labels.append("Total Invierno Base")
              diff_values.append(total_invierno_comp)
              diff_measures.append('absolute') # The base is an absolute value
              diff_text_values.append(f"${total_invierno_comp:,.0f}")

         # Add relative steps for the change in each category from Invierno to Verano
         added_comparison_diff_steps = 0 # Count significant changes added as relative steps
         for category in ordered_categories_internal_keys:
              # Only calculate difference if the category key exists in *both* dictionaries
              if category in costs_inv and category in costs_verano:
                   inv_val = costs_inv.get(category, 0.0)
                   verano_val = costs_verano.get(category, 0.0)
                   difference = verano_val - inv_val

                   if abs(difference) > 1e-9: # Only add relative step if there's a significant change
                        label = f"Cambio en {category}"
                        diff_labels.append(label)
                        diff_values.append(difference) # The difference value
                        diff_measures.append('relative') # Relative step (change)
                        sign = '+' if difference > 0 else '' # Add sign for positive values in text
                        diff_text_values.append(f"{sign}${difference:,.0f}") # Formatted text for the bar
                        added_comparison_diff_steps += 1 # Increment count

         # Add the final total bar (Total Verano) IF there were changes OR the totals are non-zero
         if abs(total_verano_comp) > 1e-9 or added_comparison_diff_steps > 0 or abs(total_invierno_comp) > 1e-9:
               # Only add the final 'Total Verano' if there was a base bar (meaning diff_labels is not empty)
               # OR if the final total is non-zero even if the base was zero
              if diff_labels or abs(total_verano_comp) > 1e-9: # Add end bar if there's a start or a non-zero end
                   diff_labels.append("Total Verano Final") # Clear label for end total
                   diff_values.append(total_verano_comp) # Final value
                   diff_measures.append('total') # MUST be 'total' for the end bar
                   diff_text_values.append(f"${total_verano_comp:,.0f}")


         # Check if there are enough points to plot the comparison waterfall (at least base + final, or base + 1 relative)
         if len(diff_labels) > 1:
              fig_diff = go.Figure(go.Waterfall(
                 name = "Diferencia",
                 orientation = "v",
                 measure = diff_measures,
                 x = diff_labels,
                 y = diff_values,
                 textposition = "outside",
                 text = diff_text_values,
                 connector = {"line": {"color": "rgb(63, 63, 63)"}},
                 # Optional: color increasing/decreasing steps (changes)
                 increasing = {"marker":{"color": "forestgreen"}},
                 decreasing = {"marker":{"color": "crimson"}},
                 # Style for the 'absolute'/'total' bars (Start/End totals)
                 totals = {"marker": {"color": "darkblue", "line": {"color": "darkblue", "width": 2}}} # Corrected parameter name and structure
              ))

              fig_diff.update_layout(
                 title = "Gráfico de Cascada - Comparación de Costos: Invierno vs. Verano",
                 showlegend = False,
                 yaxis_title="Monto ($)",
                 margin=dict(l=20, r=20, t=40, b=20),
                 hovermode="x unified",
              )
              st.plotly_chart(fig_diff, use_container_width=True)

         # Provide informative messages for edge cases where data exists but comparison isn't drawn
         elif has_both_seasons_data_parsed:
              if abs(total_invierno_comp - total_verano_comp) < 1e-9 and not category_differences_exist_in_comparison:
                   st.info(f"El Total Invierno (${total_invierno_comp:,.2f}) es igual al Total Verano (${total_verano_comp:,.2f}). No se encontraron diferencias significativas a nivel de estas categorías principales para graficar un desglose del cambio.")
              elif abs(total_invierno_comp) > 1e-9 or abs(total_verano_comp) > 1e-9: # Data present and non-zero totals, but no category diff or total diff.
                  st.info("Datos de ambas temporadas presentes. Los totales pueden ser diferentes o iguales, pero no se detectaron cambios significativos en las categorías principales ('Movilización', 'Costos Directos', etc.) para graficar un desglose del cambio.")
                  st.write(f"Total Invierno: ${total_invierno_comp:,.2f}")
                  st.write(f"Total Verano: ${total_verano_comp:,.2f}")
              else: # Both totals are zero or insignificant, and no category differences
                   st.info("Datos de ambas temporadas presentes, pero los totales y las diferencias por categoría son cero o insignificantes.")


    # Message when eco_data is a dictionary, but data for both seasons or relevant data is missing
    # Check if eco_data is a dictionary before checking its structure keys
    elif isinstance(eco_data, dict) and (eco_data.get('invierno') is not None or eco_data.get('verano') is not None):
         st.info("Archivo de presupuesto simple cargado y procesado. Para la comparación Invierno vs. Verano, verifica que el archivo contenga filas para las categorías esperadas ('Movilizacion', 'CostosDirectos', etc.) y que tengan valores no cero para ambas temporadas, o al menos en una temporada con un total diferente de cero.")

    # Message when no eco file is loaded or parsing failed critically (eco_data is the default structured dict with zeros)
    else: # eco_data is the default zeroed structure, indicating no file loaded or critical parse error
        st.info("Por favor, carga el archivo de `presupuesto_simple.csv` para generar los gráficos de cascada de costos.")

    st.markdown("---")
    st.subheader("Sugerencias para Gráficos de Cascada Adicionales")
    st.warning("Con el archivo de presupuesto simple actual, solo se pueden generar los gráficos de desglose y comparación entre temporadas. Para 'Presupuesto vs. Real' o 'Costos Históricos', necesitarías datos adicionales.")
    st.markdown("""
        - **Datos de Costos Reales:** Un archivo simple similar con los costos **REALES** incurridos para la comparación "Presupuesto vs. Real".
        - **Costos Históricos por Período:** Archivos de presupuesto (y/o reales) para cada período o un archivo consolidado con una columna 'Período' para el análisis histórico.
    """)


# --- TAB: Gestión de Stock ---
with tab_stock:
    st.header("Gestión de Stock (Catálogo e Inventario)")

    # Check if stock_df is a DataFrame and is loaded (can be empty but structured)
    if isinstance(stock_df, pd.DataFrame): # Check if parsing was successful at all (returned a DF)
        if not stock_df.empty: # Check if the DF has rows

            st.subheader("Catálogo de Items")
            # Define columns to display for the basic catalog, filter by presence in DF
            catalog_cols = ['Codigo', 'Producto', 'Categoria', 'Ubicacion']
            catalog_cols_present = [col for col in catalog_cols if col in stock_df.columns]
            if catalog_cols_present:
                 st.dataframe(stock_df[catalog_cols_present].reset_index(drop=True), use_container_width=True, hide_index=True)
            else:
                st.info("Columnas básicas de catálogo ('Codigo', 'Producto', 'Categoria', 'Ubicacion') no encontradas.")


            st.markdown("---")

            # --- Stock Inventory Section ---
            st.subheader("Inventario y Valores de Stock")

            # Check if the DataFrame has the necessary columns for inventory value calculations and STOCK_MINIMO
            # Use these checks to control what is displayed in this section
            required_inventory_value_cols = ['CantidadActual', 'CostoUnitario', 'Valor Total Item'] # 'Valor Total Item' is calculated
            has_all_required_value_cols = all(col in stock_df.columns for col in required_inventory_value_cols)
            has_stock_min_col = 'STOCK_MINIMO' in stock_df.columns


            if has_all_required_value_cols:
                 st.info("Mostrando datos de inventario (Cantidad, Costo, Valor Total, Stock Mínimo si está presente) solo para ítems con Cantidad o Valor no cero.")

                 # Filter to show items where the calculated value or quantity is non-zero for display in inventory table
                 inventory_display_df = stock_df[(abs(stock_df['CantidadActual']) > 1e-9) | (abs(stock_df['Valor Total Item']) > 1e-9)].copy()

                 if not inventory_display_df.empty:
                     # Define columns for the inventory table display, include STOCK_MINIMO if present
                     inventory_display_cols = ['Codigo', 'Producto', 'Categoria', 'CantidadActual']
                     if has_stock_min_col: inventory_display_cols.append('STOCK_MINIMO')
                     inventory_display_cols.extend(['CostoUnitario', 'Valor Total Item', 'Ubicacion'])
                     # Filter columns again to ensure they are present in the actual dataframe
                     inventory_display_cols_present = [col for col in inventory_display_cols if col in inventory_display_df.columns]

                     if inventory_display_cols_present:
                          # --- Display Inventory Table with potential styling note ---
                          # Styling for low stock is complex with st.dataframe. Provide info text instead.
                          if has_stock_min_col and 'STOCK_MINIMO' in inventory_display_cols_present:
                              if inventory_display_df['STOCK_MINIMO'].dropna().sum() > 1e-9:
                                  st.warning("Las filas con Cantidad Actual por debajo del Stock Mínimo definido se deberían resaltar (funcionalidad de resaltado de tabla requiere implementación avanzada).")
                              else:
                                  st.info("La columna 'STOCK_MINIMO' fue encontrada, pero no se definieron valores positivos de stock mínimo.")
                          else:
                              st.info("La columna 'STOCK_MINIMO' no fue encontrada en el archivo, por lo que el análisis de stock mínimo no está disponible.")

                          # Display DataFrame
                          st.dataframe(
                               inventory_display_df[inventory_display_cols_present].sort_values(by='Valor Total Item', ascending=False).reset_index(drop=True),
                               use_container_width=True,
                               hide_index=True
                          )

                     else:
                          st.warning("Columnas necesarias para mostrar el inventario detallado ('CantidadActual', 'CostoUnitario', 'Valor Total Item', etc.) faltantes después del procesamiento.")


                     st.markdown("---")
                     # --- Stock Value by Category (Pie Chart) ---
                     st.subheader("Distribución del Valor Total de Stock por Categoría")

                     # Check if required columns for this plot are present and data exists
                     if 'Categoria' in stock_df.columns and 'Valor Total Item' in stock_df.columns:
                          stock_df_for_chart = stock_df.copy() # Use stock_df (with all items) for category total calculation
                          # Grouping by category, ensure category is a clean string and handle NaNs/empty
                          stock_df_for_chart['Categoria_Group'] = stock_df_for_chart['Categoria'].astype(str).str.strip().fillna('Sin Categoría')

                          # Calculate sum of Valor Total Item by Category Group
                          stock_value_by_category = stock_df_for_chart.groupby('Categoria_Group')['Valor Total Item'].sum().reset_index()

                          # Filter out categories with zero total value for the chart, unless *all* categories have zero value and there's only one entry (like 'Sin Categoría' = 0)
                          # Keep a single category with 0 if it's the only one to show a blank/zero chart explicitly
                          if stock_value_by_category.shape[0] > 1 or (stock_value_by_category.shape[0] == 1 and abs(stock_value_by_category['Valor Total Item'].iloc[0]) > 1e-9): # Check absolute value
                              stock_value_by_category = stock_value_by_category[stock_value_by_category['Valor Total Item'].abs() > 1e-9] # Filter based on absolute value for robustness

                          if not stock_value_by_category.empty:
                               fig_stock_value_by_category = px.pie(
                                   stock_value_by_category,
                                   values='Valor Total Item',
                                   names='Categoria_Group',
                                   title='Distribución del Valor Total de Stock por Categoría',
                                   hole=0.4 # Make it a donut chart
                               )
                               st.plotly_chart(fig_stock_value_by_category, use_container_width=True)
                           # Else: no categories with non-zero value after filtering, or chart wouldn't make sense
                          else:
                               st.info("No hay categorías con un valor total de stock significativo (distinto de cero).")

                      # Display overall grand total stock value
                     grand_total_stock_value = stock_df['Valor Total Item'].sum() # Use stock_df (all items) for total value sum
                     if abs(grand_total_stock_value) > 1e-9:
                         st.subheader(f"Valor Total Estimado del Inventario: ${grand_total_stock_value:,.2f}")
                     elif not inventory_display_df.empty:
                          # If some items have non-zero quantity/value individually but the grand total sums to zero (e.g., pos/neg items)
                          st.info("El valor total calculado del inventario suma a cero o es insignificante.")
                     else:
                          st.info("No hay ítems en el inventario con cantidad o valor no cero.")

                 else:
                      st.info("El archivo de Stock cargado contiene columnas de inventario, pero todas las filas tienen Cantidad Actual y/o Costo Unitario en cero o valores insignificantes.")
                      st.write("Asegúrate de que los valores en 'CantidadActual', 'CostoUnitario' y 'STOCK_MINIMO' sean numéricos y no estén en cero en las filas de datos relevantes para que aparezcan en el inventario.")


            else: # Inventory calculation columns ('CantidadActual', 'CostoUnitario') missing after parsing
                st.info("El archivo de Stock cargado solo contiene el catálogo básico de ítems. Faltan las columnas 'CantidadActual' o 'CostoUnitario' necesarias para calcular y mostrar el valor del inventario.")


            st.markdown("---")
            # --- Missing Functionality Message ---
            st.subheader("Sugerencias para una Gestión de Stock Adicional")
            st.warning("Para una gestión de stock más completa se necesitan:")
            st.markdown("""
                - **Registro de Movimientos:** Un archivo o sistema separado para registrar entradas y salidas con fecha, tipo de movimiento ('Entrada', 'Salida', 'Ajuste'), motivo (compra, consumo, venta, merma, ajuste) y el equipo/proyecto/ubicación asociado a la salida.
                - **Reportes de Movimiento:** Tablas y gráficos del historial de uso/consumo/entrada de ítems, filtrable por fecha, ítem, equipo, etc.
                - **Evolución del Stock:** Gráficos que muestren cambios en la cantidad/valor de ítems críticos a lo largo del tiempo (requiere datos históricos de stock o registros de movimiento).
            """)


        # --- Handle case where Stock file was loaded but is empty after processing ---
        elif stock_df.empty:
             st.info("El archivo de Stock fue procesado, pero no contiene ninguna fila de datos válida (además de la cabecera).")
             st.warning("Asegúrate de que el archivo `stock_simple.csv` tenga filas debajo de la cabecera principal que no estén vacías o mal formadas, y que incluyan al menos un `Codigo` y una `Categoria` válidos.")

    # --- Handle case where Stock file was NOT loaded or parsing failed critically ---
    else: # stock_df is None (shouldn't happen with the new retrieval logic, but safety)
        st.info("Por favor, carga el archivo `stock_simple.csv` para ver el catálogo y gestión de stock.")


    # --- Stock de Combustible Section ---
    st.markdown("---")
    st.subheader("Stock de Combustible (Cantidades en Litros)")

    # Check if fuel data was loaded at all (either DF is structured or initial stock dict is present)
    # fuel_df is guaranteed to be a DataFrame (potentially empty but structured) or None on critical error
    # fuel_initial_stock is guaranteed to be a dict (potentially zeroed) or None on critical error
    has_fuel_data_available = (isinstance(fuel_df, pd.DataFrame)) or (isinstance(fuel_initial_stock, dict))

    if has_fuel_data_available:
        # Get initial stock values, defaulting to 0.0 if fuel_initial_stock is None or key is missing
        initial_gasoil = fuel_initial_stock.get('GASOIL', 0.0) if isinstance(fuel_initial_stock, dict) else 0.0
        initial_nafta = fuel_initial_stock.get('NAFTA', 0.0) if isinstance(fuel_initial_stock, dict) else 0.0

        total_ingress_gasoil = 0.0
        total_egress_gasoil = 0.0
        total_ingress_nafta = 0.0
        total_egress_nafta = 0.0

        # Check if fuel_df is usable for calculations (is DataFrame, not empty, has essential columns)
        required_fuel_sum_cols = ['Tipo de Comb.', 'Lts Ingreso', 'Lts Egreso']
        if isinstance(fuel_df, pd.DataFrame) and not fuel_df.empty and all(col in fuel_df.columns for col in required_fuel_sum_cols):
             # Filter data by fuel type
             df_gasoil = fuel_df[fuel_df['Tipo de Comb.'] == 'GASOIL'].copy() # Use .copy()
             df_nafta = fuel_df[fuel_df['Tipo de Comb.'] == 'NAFTA'].copy()   # Use .copy()

             # Calculate sums. Use .sum() which correctly handles empty DFs or columns (results in 0.0).
             total_ingress_gasoil = df_gasoil['Lts Ingreso'].sum() if 'Lts Ingreso' in df_gasoil.columns else 0.0
             total_egress_gasoil = df_gasoil['Lts Egreso'].sum() if 'Lts Egreso' in df_gasoil.columns else 0.0
             total_ingress_nafta = df_nafta['Lts Ingreso'].sum() if 'Lts Ingreso' in df_nafta.columns else 0.0
             total_egress_nafta = df_nafta['Lts Egreso'].sum() if 'Lts Egreso' in df_nafta.columns else 0.0
        elif isinstance(fuel_df, pd.DataFrame) and (fuel_df.empty or not all(col in fuel_df.columns for col in required_fuel_sum_cols)):
             # fuel_df is a DataFrame but is empty or missing columns required for sum.
             missing_sum_cols = [col for col in required_fuel_sum_cols if col not in fuel_df.columns]
             if missing_sum_cols:
                 st.warning(f"Stock Combustible: Faltan columnas ({', '.join(missing_sum_cols)}) en el DataFrame de combustible procesado para calcular totales de ingresos/egresos.")
             else:
                 st.info("Stock Combustible: El archivo de Combustible no contiene registros de movimientos de datos válidos.")


        # Calculate estimated current stock based on initial balance + ingress - egress
        current_stock_gasoil = initial_gasoil + total_ingress_gasoil - total_egress_gasoil
        current_stock_nafta = initial_nafta + total_ingress_nafta - total_egress_nafta


        # Determine if we have *any* data to display for Gasoil or Nafta stock breakdown
        has_gasoil_data_to_display = abs(initial_gasoil) > 1e-9 or abs(total_ingress_gasoil) > 1e-9 or abs(total_egress_gasoil) > 1e-9 or abs(current_stock_gasoil) > 1e-9
        has_nafta_data_to_display = abs(initial_nafta) > 1e-9 or abs(total_ingress_nafta) > 1e-9 or abs(total_egress_nafta) > 1e-9 or abs(current_stock_nafta) > 1e-9


        if has_gasoil_data_to_display:
            st.write(f"- Saldo Inicial GASOIL (línea SETUP): {initial_gasoil:,.2f} Lts")
            st.write(f"- Ingreso Total GASOIL (registrado en DATA): {total_ingress_gasoil:,.2f} Lts")
            st.write(f"- Egreso Total GASOIL (registrado en DATA): {total_egress_gasoil:,.2f} Lts")
            st.write(f"**Saldo Actual Estimado GASOIL: {current_stock_gasoil:,.2f} Lts**")
            if has_nafta_data_to_display: st.markdown("---") # Add separator only if displaying both types

        if has_nafta_data_to_display:
            st.write(f"- Saldo Inicial NAFTA (línea SETUP): {initial_nafta:,.2f} Lts")
            st.write(f"- Ingreso Total NAFTA (registrado en DATA): {total_ingress_nafta:,.2f} Lts")
            st.write(f"- Egreso Total NAFTA (registrado en DATA): {total_egress_nafta:,.2f} Lts")
            st.write(f"**Saldo Actual Estimado NAFTA: {current_stock_nafta:,.2f} Lts**")


        if not has_gasoil_data_to_display and not has_nafta_data_to_display:
             st.info("No se encontraron datos de ingresos/egresos no cero ni saldos iniciales no cero para combustible.")


        st.warning("Este cálculo de stock de combustible es una estimación basada en los saldos iniciales SETUP y los ingresos/egresos registrados en las filas de datos. Las filas de egreso marcadas como 'PRESTAMO' en los comentarios NO se excluyen automáticamente de los egresos totales en este cálculo simple.")

        st.markdown("---")
        st.subheader("Sugerencias para Mejorar la Gestión de Stock de Combustible")
        st.markdown("""
            - Asegúrate de que las líneas `SETUP` para saldos iniciales y mapeos de código estén correctas y que los campos numéricos tengan valores válidos.
            - Considera usar una columna `TipoMovimiento` ('Consumo', 'Prestamo', 'Devolucion', 'Ajuste Entrada', 'Ajuste Salida') en las filas de datos para una gestión más clara y controlable. Si marcas un egreso como 'Prestamo', podrías querer excluirlo del cálculo de 'Egreso Total'.
            - Un registro periódico de saldo a fecha (`DATA, FechadeConteo, SaldoObservado,...`) ayudaría a auditar y ajustar el saldo calculado con el saldo real.
            - Graficar la evolución del saldo de combustible a lo largo del tiempo sería útil (requiere los movimientos con fecha o saldos periódicos).
        """)

    # Handle cases where Fuel file was processed but has specific emptiness states
    # Check if fuel_df is a DataFrame before checking its emptiness
    elif isinstance(fuel_df, pd.DataFrame) and fuel_df.empty and isinstance(fuel_initial_stock, dict) and (abs(fuel_initial_stock.get('GASOIL', 0.0)) > 1e-9 or abs(fuel_initial_stock.get('NAFTA', 0.0)) > 1e-9):
         st.info(f"El archivo de Combustible fue procesado. Se encontraron saldos iniciales en las líneas SETUP, pero no registros de movimientos (ingresos/egresos) válidos en las filas DATA.")
         st.write(f"Saldo Inicial GASOIL: {fuel_initial_stock.get('GASOIL', 0.0):,.2f} Lts")
         st.write(f"Saldo Inicial NAFTA: {fuel_initial_stock.get('NAFTA', 0.0):,.2f} Lts")
         st.warning("Asegúrate de que haya filas debajo de la cabecera principal ('Fecha,Equipo,...') que contengan valores válidos en 'LtsIngreso'/'LtsEgreso' y demás columnas.")

    elif isinstance(fuel_df, pd.DataFrame) and fuel_df.empty and isinstance(fuel_initial_stock, dict) and not (abs(fuel_initial_stock.get('GASOIL', 0.0)) > 1e-9 or abs(fuel_initial_stock.get('NAFTA', 0.0)) > 1e-9):
         st.info("El archivo de Combustible fue procesado. No se encontraron registros de movimientos válidos (DATA) ni saldos iniciales no cero (SETUP).")

    # Handle case where Fuel file was NOT loaded or parsing failed critically (fuel_df is None)
    else: # fuel_df is None (shouldn't happen with the new retrieval logic, but safety)
        st.info("Por favor, carga el archivo `combustible_simple_v2.csv` para ver el stock de combustible y su gestión.")


# --- TAB: Creación de Presupuestos ---
with tab_budget:
    st.header("Funcionalidad Sugerida: Creación de Presupuestos / Cotizaciones")

    st.info("El propósito de la aplicación actual es principalmente analizar archivos CSV existentes (Stock, Presupuesto simple, Combustible). **Esta sección no implementa la creación dinámica de nuevos presupuestos.** ")
    st.warning("Desarrollar una funcionalidad completa para crear presupuestos/cotizaciones seleccionando elementos (servicios, equipos, materiales de stock) requeriría la implementación de lógica de precios compleja y bases de datos maestras.")

    st.subheader("Información Necesaria Típicamente para Crear Presupuestos Dinámicos")
    st.markdown("""
        Para crear presupuestos y cotizaciones seleccionando servicios, equipos y materiales de forma dinámica, generalmente se necesitan bases de datos maestras detalladas, como:
        - **Maestro de Servicios/Productos para Venta:** Listado de ítems o servicios que se cotizan al cliente, su descripción detallada y, crucialmente, el **precio unitario de cotización (precio de venta)**.
        - **Maestro de Costos de Recursos:** Listados separados con los costos unitarios internos de diferentes tipos de recursos, como:
            - Tarifas horarias o diarias de **Mano de Obra** (por categoría/especialidad).
            - Costos horarios o por kilómetro/uso de **Equipos Propios**.
            - Costos unitarios de **Materiales de Stock** (esto lo podrías obtener de tu archivo de stock cargado si tiene el campo `CostoUnitario`).
            - Costos de Subcontratistas, Permisos, etc.
        - **Estructura/Reglas de Costos/Precio:** Lógica para calcular automáticamente o ingresar manualmente otros costos no directos (costos indirectos, gastos generales, gastos de venta) y la **Utilidad** deseada. Finalmente, considerar Impuestos (IVA, etc.).

        En la estructura simple de presupuesto que se carga (presupuesto_simple.csv), solo defines totales agrupados por unas pocas categorías (`Movilizacion`, `Costos Directos`, etc.), que son útiles para el análisis a alto nivel o gráficos de cascada, pero no contienen el detalle granular necesario para armar una cotización ítem por ítem.
    """)

    # --- Display Example Structure based on loaded files ---
    # Check if eco_data is a dictionary before trying to access keys
    if isinstance(eco_data, dict) and (eco_data.get('invierno') is not None or eco_data.get('verano') is not None):
         st.subheader("Estructura de Costos de Ejemplo (obtenida del archivo `presupuesto_simple.csv`):")
         st.write("Esta estructura muestra los totales agrupados por las categorías principales:")
         col1, col2 = st.columns(2)
         with col1:
              st.markdown("##### Invierno")
              # Iterate over keys present in the dict, using the ordered list for preference
              for key in [k for k in ordered_categories_internal_keys if k in eco_data.get('invierno', {})]:
                   value = eco_data.get('invierno', {}).get(key, 0.0)
                   st.write(f"- **{key}**: ${value:,.2f}")
              # Handle any extra keys that might have been parsed but aren't in the ordered list
              extra_inv_keys = [k for k in eco_data.get('invierno', {}).keys() if k not in ordered_categories_internal_keys]
              if extra_inv_keys:
                   st.markdown("###### Otras Categorías (Invierno):")
                   for key in extra_inv_keys:
                        value = eco_data.get('invierno', {}).get(key, 0.0)
                        st.write(f"- **{key}**: ${value:,.2f}")


         with col2:
              st.markdown("##### Verano")
              # Iterate over keys present in the dict, using the ordered list for preference
              for key in [k for k in ordered_categories_internal_keys if k in eco_data.get('verano', {})]:
                   value = eco_data.get('verano', {}).get(key, 0.0)
                   st.write(f"- **{key}**: ${value:,.2f}")
               # Handle any extra keys that might have been parsed but aren't in the ordered list
              extra_verano_keys = [k for k in eco_data.get('verano', {}).keys() if k not in ordered_categories_internal_keys]
              if extra_verano_keys:
                   st.markdown("###### Otras Categorías (Verano):")
                   for key in extra_verano_keys:
                        value = eco_data.get('verano', {}).get(key, 0.0)
                        st.write(f"- **{key}**: ${value:,.2f}")


    # Check if eco_data is a dictionary but appears empty (no 'invierno' or 'verano' keys with data)
    elif isinstance(eco_data, dict) and (eco_data.get('invierno') is None and eco_data.get('verano') is None):
         st.info("Archivo de Presupuesto cargado, pero no se extrajeron datos de costos con las estructuras de 'invierno'/'verano' esperadas.")
    # This condition should now cover the case where eco_data is the zeroed structure returned by the parser
    elif isinstance(eco_data, dict) and (eco_data.get('invierno') is not None and eco_data.get('verano') is not None) and all(abs(eco_data.get('invierno', {}).get(k, 0.0)) < 1e-9 for k in ordered_categories_internal_keys) and all(abs(eco_data.get('verano', {}).get(k, 0.0)) < 1e-9 for k in ordered_categories_internal_keys):
         st.info("Archivo de Presupuesto cargado, pero los costos para las categorías principales son cero o insignificantes en ambas temporadas.")

    # Message when no eco file is loaded or parsing failed critically (eco_data is the default structured dict with zeros)
    else: # eco_data is the default zeroed structure, indicating no file loaded or critical parse error
        st.info("Carga el archivo `presupuesto_simple.csv` para ver la estructura de costos de ejemplo leída.")


    if isinstance(stock_df, pd.DataFrame) and not stock_df.empty:
         st.subheader("Primeros Ítems del Catálogo de Stock (obtenidos del archivo `stock_simple.csv`):")
         # Check if the stock file included CostoUnitario (which is key for material costs in a budget)
         has_costo_unitario_col_in_df = 'CostoUnitario' in stock_df.columns and not stock_df['CostoUnitario'].replace([0.0], np.nan).isnull().all() # Check column exists and has at least one non-zero, non-null value

         # Define display columns for the stock example
         display_cols_stock_example = ['Codigo', 'Producto', 'Categoria', 'CantidadActual'] # Base columns
         if 'CostoUnitario' in stock_df.columns: display_cols_stock_example.append('CostoUnitario') # Add costo unitario if present
         if 'Ubicacion' in stock_df.columns: display_cols_stock_example.append('Ubicacion') # Add ubicacion if present

         display_cols_stock_example_present = [col for col in display_cols_stock_example if col in stock_df.columns] # Filter by actual presence

         if display_cols_stock_example_present:
              if has_costo_unitario_col_in_df:
                   st.write("Este catálogo (si tiene la columna `CostoUnitario` completa) podría servir como un **maestro de materiales** para presupuestar costos:")
              else:
                   st.write("Este catálogo básico es útil para referencia, pero **le falta la columna `CostoUnitario` con valores numéricos** para ser usado directamente en la creación de presupuestos que calculen costos de materiales.")

              # Display first few rows of the stock data
              st.dataframe(stock_df[display_cols_stock_example_present].head(10).reset_index(drop=True), use_container_width=True, hide_index=True)
         else:
              st.info("El archivo de Stock cargado no contiene columnas básicas ('Codigo', 'Producto', 'Categoria') para mostrar un catálogo de ejemplo.")

    elif isinstance(stock_df, pd.DataFrame) and stock_df.empty:
        st.info("Archivo de Stock cargado y procesado, pero está vacío.")

    # Message when no stock file is loaded or parsing failed critically (stock_df is the default structured DF)
    else: # stock_df is the default structured DF, indicating no file loaded or critical parse error
        st.info("Carga el archivo `stock_simple.csv` para ver un catálogo de ítems de ejemplo (potencial maestro de materiales).")

    st.markdown("---")
    st.subheader("Desarrollo de Funcionalidad Futura Sugerida:")
    st.markdown("""
        - Implementar una sección de "Nuevo Presupuesto" donde se puedan:
            - Ingresar detalles generales (Nombre, Cliente, Fecha).
            - Añadir ítems seleccionándolos de los "Maestros" (que cargarías o definirías): Servicios, Equipos, Materiales de Stock.
            - Para cada ítem seleccionado, especificar cantidad, unidades, y si es necesario, ajustar el costo/precio unitario para esta cotización particular.
            - Calcular totales: Subtotal por categorías, Costo Total Directo, Costo Total Indirecto, Utilidad, Precio Total de Venta.
        - Requeriría definir cómo se calculan los costos indirectos y la utilidad (ej: porcentaje sobre costos directos, monto fijo, etc.).
        - Generar un resumen o un reporte de presupuesto básico.
    """)


# --- Instructions for File Modification in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("Formatos de Archivos CSV Simples Recomendados")
st.sidebar.info("Para el correcto funcionamiento, carga archivos CSV con estas estructuras exactas:")
st.sidebar.markdown("""
- **Configuración General de CSV:** Delimitado por comas (`,` ), guardado con codificación **UTF-8 sin BOM**. Usar **punto `.`** como separador decimal. No usar separadores de miles (ej. `1250.50`, no `1.250,50`). Si un campo de texto contiene comas o comillas literales, el campo debe ir encerrado entre comillas dobles (`"`). Las filas vacías serán ignoradas.

- **Stock (`stock_simple.csv`):**
    - **Cabecera Exacta:** `Codigo,Producto,Categoria,CantidadActual,CostoUnitario,Ubicacion,STOCK_MINIMO`
    - **Datos:** 7 campos por línea (separados por comas), comenzando después de la fila de cabecera. `Codigo` y `Categoria` no deben estar vacíos para que la fila sea considerada un ítem válido. `CantidadActual`, `CostoUnitario`, y `STOCK_MINIMO` deben ser números.
    - **Ejemplo Fila de Datos:** `ABC-123,Martillo Grande,HERRAMIENTAS MANUALES,5.0,1250.50,Almacén 1,3.0` (Puedes dejar `STOCK_MINIMO` vacío si no aplica o si no tienes el dato; el parser lo tratará como 0).

- **Presupuesto (`presupuesto_simple.csv`):**
    - **Cabecera Exacta:** `Categoria,Invierno,Verano`
    - **Datos:** 3 campos por línea, comenzando después de la cabecera. `Categoria` debe ser una de las esperadas (`Movilizacion`, `CostosDirectos`, `CostosIndirectosGenerales`, `Utilidades`). `Invierno` y `Verano` deben ser números.
    - **Ejemplo Fila de Datos:** `CostosDirectos,388289413.85,180310026.04`

- **Combustible (`combustible_simple_v2.csv`):**
    - **Líneas de Configuración (opcionales, al inicio):** `SETUP,Saldo Inicial GASOIL,NUMERO_SALDO` ; `SETUP,Saldo Inicial NAFTA,NUMERO_SALDO` ; `SETUP,MAPEO CODIGO,CODIGO_COMBUSTIBLE,TIPO_COMBUSTIBLE`. (Ej: `SETUP,Saldo Inicial GASOIL,500.00` ; `SETUP,MAPEO CODIGO,001,GASOIL`). Puedes tener múltiples líneas MAPEO CODIGO.
    - **Cabecera de Datos Exacta:** `Fecha,Equipo,CodigoCombustible,LtsIngreso,LtsEgreso,HsKm,Comentarios,CostoUnitarioLt`
    - **Datos:** 8 campos por línea, comenzando DESPUÉS de la línea de Cabecera de Datos.
        - `Fecha`: Formato fecha (ej. `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`). Se espera día primero (`dayfirst`).
        - `Equipo`, `CodigoCombustible`, `Comentarios`: Texto (puede ser vacío `HsKm` para Consumos con Uso o `0.0` si aplica.
        - `LtsIngreso`, `LtsEgreso`, `CostoUnitarioLt`: Números (pueden ser `0.0`). Para ingresos, `LtsIngreso > 0`, `LtsEgreso = 0`. Para egresos (consumos), `LtsIngreso = 0`, `LtsEgreso > 0`. Para `CostoUnitarioLt`, ingresa el costo por litro para los INGRESOS.
    - **Ejemplo Fila de Datos:** `01-03-2025,MO-13-V,001,0.0,98.0,1778.0,Consumo normal en terreno,1200.50` (Egreso de 98Lts GASOIL (código 001) en equipo MO-13-V, a 1200.50/Lt, con 1778 HsKm, comentario 'Consumo normal'). Nota: el CostoUnitarioLt en una fila de egreso en este formato es para asociar el costo unitario del *lote egresado* si no tienes sistema de precios promedio/FIFO. Este análisis simple usa `LtsEgreso * CostoUnitarioLt` para el costo total del movimiento.

""")
st.sidebar.markdown("---")

