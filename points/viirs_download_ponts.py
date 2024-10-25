import requests
import json
import os
import calendar

# Retrieve access token
params = {    
    'client_id': 'eogdata_oidc',
    'client_secret': '2677ad81-521b-4869-8480-6d05b9e57d48',
    'username': '',
    'password': '',
    'grant_type': 'password'
}
token_url = 'https://eogauth.mines.edu/auth/realms/master/protocol/openid-connect/token'
response = requests.post(token_url, data = params)
access_token_dict = json.loads(response.text)
access_token = access_token_dict.get('access_token')

list_months = range(1, 13)  # Mês de 1 até 12
year = 2019  # Ano de referência

for month in list_months:
    # Obter o último dia do mês
    last_day = calendar.monthrange(year, month)[1]

    for day in range(1, last_day + 1):
        # Formatar mês e dia para ficarem com dois dígitos
        month_str = f"{month:02d}"
        day_str = f"{day:02d}"
        
        # Gerar URL
        data_url = f'https://eogdata.mines.edu/wwwdata/viirs_products/vnf/v30/rearrange/{year}/{month_str}/npp/VNF_npp_d{year}{month_str}{day_str}_noaa_v30.csv.gz'
        
        # Cabeçalhos de autenticação
        auth = 'Bearer ' + access_token
        headers = {'Authorization': auth}

        # Fazer a requisição ao servidor
        response = requests.get(data_url, headers=headers)

        # Nome do arquivo de saída
        output_file = os.path.basename(data_url)
        file_p = os.path.join("/home/marycamila/flaresat/source/viirs", output_file)

        # Salvar o conteúdo do arquivo na pasta especificada
        with open(file_p, 'wb') as f:
            f.write(response.content)