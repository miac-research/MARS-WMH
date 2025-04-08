#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def create_html_with_png(fnPNG, fnHTML, df=None, text=None):
  import base64, os, re
  import pandas as pd
  import numpy as np

  # Read CSV file, if df is not provided
  if df is None:
    fnCSV = re.sub('_QC\.html$', '.csv', fnHTML)
    if os.path.exists(fnCSV):
        df = pd.read_csv(fnCSV)
    else:
        df = None
  
  # Adjust df and transform into HTML table
  if df is None:
      dfHTML = ''
  else:
      # convert values to integer, if appropriate
      df['Value'] = df['Value'].astype('object')
      for i,row in df.iterrows():
          if np.mod(row['Value'],1)<0.001:
            df.loc[i,'Value'] = int(row['Value'])
      # set index and convert
      df.set_index('Name', inplace=True)
      dfHTML = df.to_html(classes="mystyle", index=True, index_names=False)

  # Convert text to HTML paragraph
  if text is None:
     text = ''
  else:
     text = '<br>'.join(text)
     text = f'<p style=mystyle >{text}</p>'
    #  text = f'<p>{text}</p>'

  # Read image
  with open(fnPNG, 'rb') as f:
      image_data = f.read()
      base64_image = base64.b64encode(image_data).decode('utf-8')
  
  # Style
  style='''
  <style>
    body {
      font-size: 11pt;
      font-family: Arial;
      background-color: #f4f4f4;
    }
    .mystyle {
        font-size: 11pt;
        font-family: Arial;
        border-collapse: collapse;
        border: 1px solid black;
    }
    .mystyle th:nth-child(0), th:nth-child(2), th:nth-child(3) {
        background-color: #f0f0f0;
    }
    .mystyle td, th {
        padding: 5px;
    }
        
    .mystyle tr:nth-child(even) {
        background: #f0f0f0;
    }

    .mystyle tr:nth-child(odd) {
        background: #FFF;
    }

    .mystyle tr:hover {
        background: silver;
        cursor: pointer;
    }
  </style>
  '''

  # Create html string
  html=f'''
  <!DOCTYPE html>
  <html>
    <head>
      <title>QC Image WMH Segmentation</title>
      {style}
    </head>
    <body>
      <img width=100% src="data:image/png;base64,{base64_image}">
      {text}
      {dfHTML}
    </body>
  </html>
  '''
  
  # write to HTML file
  with open(fnHTML, 'w') as fp:
      fp.write(html)