import pickle
import PySimpleGUI as sg
import cogface as cf
from Profile import Profile
import numpy as np
import matplotlib.pyplot as plt

sg.theme('Python')

layoutStart = [[sg.Button('Create New Database'), sg.Button('Load Database')],
          [sg.Button('Close')]]
windowStart = sg.Window('Cog*Face: Create or Load Database', layoutStart, size=(600, 400))
while True:
    event, values = windowStart.read()      
    if event == sg.WIN_CLOSED or event == 'Close':
        break
    if event == 'Create New Database':
        cf.db.create_database()
        break
    if event == 'Load Database':
        cf.db.load_database()
        break

windowStart.close()

if event != sg.WIN_CLOSED and event != 'Close':
    layoutHome = [[sg.Button('Add Face'), sg.Button('Detect Faces')],
                  [sg.Button('Close')]]
    windowHome = sg.Window('Cog*Face: Function Select', layoutHome, size=(600, 400))

    while True:
        event, values = windowHome.read()
        if event == sg.WIN_CLOSED or event == 'Close':
            break
        if event == 'Add Face':
            layoutAdd = [[sg.Text('Select File'), sg.InputText('Face.jpg'), sg.FileBrowse()],
                         [sg.Text('Input Person Name'), sg.InputText('Person')],
                         [sg.Button('OK'), sg.Button('Cancel')]]
            windowAdd = sg.Window('Cog*Face: Add Face', layoutAdd, size=(600, 400))
            while True:
                event, values = windowAdd.read()
                if event == sg.WIN_CLOSED or event == 'Cancel':
                    break
                if event == 'OK':
                    img = cf.upload_picture(values[0])
                    cf.db.add_image(values[1], img)
                    break
            windowAdd.close()
        if event == 'Detect Faces':
            layoutAdd = [[sg.Button('Import from Files'), sg.Button('Take a Picture')],
                         [sg.Button('Cancel')]]
            windowAdd = sg.Window('Cog*Face: Detect Face', layoutAdd, size=(600, 400))
            while True:
                event, values = windowAdd.read()
                if event == sg.WIN_CLOSED or event == 'Cancel':
                    break
                if event == 'Import from Files':
                    layoutAdd = [[sg.Text('Select File'), sg.InputText('Image.jpg'), sg.FileBrowse()],
                         [sg.Button('OK'), sg.Button('Cancel')]]
                    windowAdd = sg.Window('Cog*Face: Detect Face', layoutAdd, size=(600, 400))
                    while True:
                        event, values = windowAdd.read()
                        if event == sg.WIN_CLOSED or event == 'Cancel':
                            break
                        if event == 'OK':
                            cf.detect_from_file(values[0])
                            plt.savefig("tempimg.png")
                            layoutDisplay = [[sg.Image("tempimg.png")], [sg.Button('Close')]]
                            windowDisplay = sg.Window('Cog*Face: Detected Faces', layoutDisplay, size=(1200, 1000))
                            while True:
                                event, values = windowDisplay.read()
                                if event == sg.WIN_CLOSED or event == 'Close':
                                    break
                            windowDisplay.close()
                            break
                    windowAdd.close()
                    break
                if event == 'Take a Picture':
                    cf.detect_from_camera()
                    plt.savefig("tempimg.png")
                    layoutDisplay = [[sg.Image("tempimg.png")], [sg.Button('Close')]]
                    windowDisplay = sg.Window('Cog*Face: Detected Faces', layoutDisplay, size=(1200, 1000))
                    while True:
                        event, values = windowDisplay.read()
                        if event == sg.WIN_CLOSED or event == 'Close':
                            break
                    windowDisplay.close()
                    break
            windowAdd.close()
    windowHome.close()