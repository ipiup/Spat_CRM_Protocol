#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on juillet 02, 2026, at 09:55
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard
from psychopy.hardware.button import ButtonBox

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.3'
expName = 'CRM_Spat'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\chamery\\Documents\\Spat_CRM_Protocol\\Spat_CRM_Protocol\\src\\CRM_Spat.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "trial" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='Selectionner la réponse',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    R1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R1',
        depth=-1
    )
    R1.buttonClock = core.Clock()
    R2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R2',
        depth=-2
    )
    R2.buttonClock = core.Clock()
    R3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R3',
        depth=-3
    )
    R3.buttonClock = core.Clock()
    R4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R4',
        depth=-4
    )
    R4.buttonClock = core.Clock()
    R5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R5',
        depth=-5
    )
    R5.buttonClock = core.Clock()
    R6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(0.225, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R6',
        depth=-6
    )
    R6.buttonClock = core.Clock()
    R7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R7',
        depth=-7
    )
    R7.buttonClock = core.Clock()
    R8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R8',
        depth=-8
    )
    R8.buttonClock = core.Clock()
    B1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B1',
        depth=-9
    )
    B1.buttonClock = core.Clock()
    B2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B2',
        depth=-10
    )
    B2.buttonClock = core.Clock()
    B3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B3',
        depth=-11
    )
    B3.buttonClock = core.Clock()
    B4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B4',
        depth=-12
    )
    B4.buttonClock = core.Clock()
    B5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B5',
        depth=-13
    )
    B5.buttonClock = core.Clock()
    B6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(.225, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B6',
        depth=-14
    )
    B6.buttonClock = core.Clock()
    B7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B7',
        depth=-15
    )
    B7.buttonClock = core.Clock()
    B8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B8',
        depth=-16
    )
    B8.buttonClock = core.Clock()
    V1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V1',
        depth=-17
    )
    V1.buttonClock = core.Clock()
    V2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V2',
        depth=-18
    )
    V2.buttonClock = core.Clock()
    V3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V3',
        depth=-19
    )
    V3.buttonClock = core.Clock()
    V4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V4',
        depth=-20
    )
    V4.buttonClock = core.Clock()
    V5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V5',
        depth=-21
    )
    V5.buttonClock = core.Clock()
    V6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(.225, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V6',
        depth=-22
    )
    V6.buttonClock = core.Clock()
    V7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V7',
        depth=-23
    )
    V7.buttonClock = core.Clock()
    V8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V8',
        depth=-24
    )
    V8.buttonClock = core.Clock()
    J1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J1',
        depth=-25
    )
    J1.buttonClock = core.Clock()
    J2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J2',
        depth=-26
    )
    J2.buttonClock = core.Clock()
    J3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J3',
        depth=-27
    )
    J3.buttonClock = core.Clock()
    J4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J4',
        depth=-28
    )
    J4.buttonClock = core.Clock()
    J5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J5',
        depth=-29
    )
    J5.buttonClock = core.Clock()
    J6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(.225, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J6',
        depth=-30
    )
    J6.buttonClock = core.Clock()
    J7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J7',
        depth=-31
    )
    J7.buttonClock = core.Clock()
    J8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J8',
        depth=-32
    )
    J8.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "escu" ---
    Pasdeffort = visual.ButtonStim(win, 
        text="Pas d'effort", font='Arvo',
        pos=(0, -0.4),
        letterHeight=0.03,
        size=(0.9, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Pasdeffort',
        depth=0
    )
    Pasdeffort.buttonClock = core.Clock()
    __1 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, -0.35),
        letterHeight=0.03,
        size=(0.85, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__1',
        depth=-1
    )
    __1.buttonClock = core.Clock()
    Trespeudeffort = visual.ButtonStim(win, 
        text="Très peu d'effort", font='Arvo',
        pos=(0, -0.3),
        letterHeight=0.03,
        size=(0.8, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Trespeudeffort',
        depth=-2
    )
    Trespeudeffort.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "trial" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='Selectionner la réponse',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    R1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R1',
        depth=-1
    )
    R1.buttonClock = core.Clock()
    R2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R2',
        depth=-2
    )
    R2.buttonClock = core.Clock()
    R3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R3',
        depth=-3
    )
    R3.buttonClock = core.Clock()
    R4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R4',
        depth=-4
    )
    R4.buttonClock = core.Clock()
    R5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R5',
        depth=-5
    )
    R5.buttonClock = core.Clock()
    R6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(0.225, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R6',
        depth=-6
    )
    R6.buttonClock = core.Clock()
    R7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R7',
        depth=-7
    )
    R7.buttonClock = core.Clock()
    R8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R8',
        depth=-8
    )
    R8.buttonClock = core.Clock()
    B1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B1',
        depth=-9
    )
    B1.buttonClock = core.Clock()
    B2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B2',
        depth=-10
    )
    B2.buttonClock = core.Clock()
    B3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B3',
        depth=-11
    )
    B3.buttonClock = core.Clock()
    B4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B4',
        depth=-12
    )
    B4.buttonClock = core.Clock()
    B5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B5',
        depth=-13
    )
    B5.buttonClock = core.Clock()
    B6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(.225, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B6',
        depth=-14
    )
    B6.buttonClock = core.Clock()
    B7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B7',
        depth=-15
    )
    B7.buttonClock = core.Clock()
    B8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B8',
        depth=-16
    )
    B8.buttonClock = core.Clock()
    V1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V1',
        depth=-17
    )
    V1.buttonClock = core.Clock()
    V2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V2',
        depth=-18
    )
    V2.buttonClock = core.Clock()
    V3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V3',
        depth=-19
    )
    V3.buttonClock = core.Clock()
    V4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V4',
        depth=-20
    )
    V4.buttonClock = core.Clock()
    V5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V5',
        depth=-21
    )
    V5.buttonClock = core.Clock()
    V6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(.225, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V6',
        depth=-22
    )
    V6.buttonClock = core.Clock()
    V7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V7',
        depth=-23
    )
    V7.buttonClock = core.Clock()
    V8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V8',
        depth=-24
    )
    V8.buttonClock = core.Clock()
    J1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J1',
        depth=-25
    )
    J1.buttonClock = core.Clock()
    J2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J2',
        depth=-26
    )
    J2.buttonClock = core.Clock()
    J3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J3',
        depth=-27
    )
    J3.buttonClock = core.Clock()
    J4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J4',
        depth=-28
    )
    J4.buttonClock = core.Clock()
    J5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J5',
        depth=-29
    )
    J5.buttonClock = core.Clock()
    J6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(.225, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J6',
        depth=-30
    )
    J6.buttonClock = core.Clock()
    J7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J7',
        depth=-31
    )
    J7.buttonClock = core.Clock()
    J8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J8',
        depth=-32
    )
    J8.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "escu" ---
    Pasdeffort = visual.ButtonStim(win, 
        text="Pas d'effort", font='Arvo',
        pos=(0, -0.4),
        letterHeight=0.03,
        size=(0.9, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Pasdeffort',
        depth=0
    )
    Pasdeffort.buttonClock = core.Clock()
    __1 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, -0.35),
        letterHeight=0.03,
        size=(0.85, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__1',
        depth=-1
    )
    __1.buttonClock = core.Clock()
    Trespeudeffort = visual.ButtonStim(win, 
        text="Très peu d'effort", font='Arvo',
        pos=(0, -0.3),
        letterHeight=0.03,
        size=(0.8, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Trespeudeffort',
        depth=-2
    )
    Trespeudeffort.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "trial" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='Selectionner la réponse',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    R1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R1',
        depth=-1
    )
    R1.buttonClock = core.Clock()
    R2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R2',
        depth=-2
    )
    R2.buttonClock = core.Clock()
    R3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R3',
        depth=-3
    )
    R3.buttonClock = core.Clock()
    R4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R4',
        depth=-4
    )
    R4.buttonClock = core.Clock()
    R5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R5',
        depth=-5
    )
    R5.buttonClock = core.Clock()
    R6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(0.225, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R6',
        depth=-6
    )
    R6.buttonClock = core.Clock()
    R7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R7',
        depth=-7
    )
    R7.buttonClock = core.Clock()
    R8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, -.075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, -1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='R8',
        depth=-8
    )
    R8.buttonClock = core.Clock()
    B1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B1',
        depth=-9
    )
    B1.buttonClock = core.Clock()
    B2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B2',
        depth=-10
    )
    B2.buttonClock = core.Clock()
    B3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B3',
        depth=-11
    )
    B3.buttonClock = core.Clock()
    B4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B4',
        depth=-12
    )
    B4.buttonClock = core.Clock()
    B5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B5',
        depth=-13
    )
    B5.buttonClock = core.Clock()
    B6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(.225, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B6',
        depth=-14
    )
    B6.buttonClock = core.Clock()
    B7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B7',
        depth=-15
    )
    B7.buttonClock = core.Clock()
    B8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, .225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, -1.0000, 1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='B8',
        depth=-16
    )
    B8.buttonClock = core.Clock()
    V1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V1',
        depth=-17
    )
    V1.buttonClock = core.Clock()
    V2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V2',
        depth=-18
    )
    V2.buttonClock = core.Clock()
    V3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V3',
        depth=-19
    )
    V3.buttonClock = core.Clock()
    V4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V4',
        depth=-20
    )
    V4.buttonClock = core.Clock()
    V5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V5',
        depth=-21
    )
    V5.buttonClock = core.Clock()
    V6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(.225, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V6',
        depth=-22
    )
    V6.buttonClock = core.Clock()
    V7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V7',
        depth=-23
    )
    V7.buttonClock = core.Clock()
    V8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, -.225),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(-1.0000, 0.0039, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='V8',
        depth=-24
    )
    V8.buttonClock = core.Clock()
    J1 = visual.ButtonStim(win, 
        text='1', font='Arvo',
        pos=(-.525, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J1',
        depth=-25
    )
    J1.buttonClock = core.Clock()
    J2 = visual.ButtonStim(win, 
        text='2', font='Arvo',
        pos=(-.375, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J2',
        depth=-26
    )
    J2.buttonClock = core.Clock()
    J3 = visual.ButtonStim(win, 
        text='3', font='Arvo',
        pos=(-.225, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J3',
        depth=-27
    )
    J3.buttonClock = core.Clock()
    J4 = visual.ButtonStim(win, 
        text='4', font='Arvo',
        pos=(-.075, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J4',
        depth=-28
    )
    J4.buttonClock = core.Clock()
    J5 = visual.ButtonStim(win, 
        text='5', font='Arvo',
        pos=(.075, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J5',
        depth=-29
    )
    J5.buttonClock = core.Clock()
    J6 = visual.ButtonStim(win, 
        text='6', font='Arvo',
        pos=(.225, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J6',
        depth=-30
    )
    J6.buttonClock = core.Clock()
    J7 = visual.ButtonStim(win, 
        text='7', font='Arvo',
        pos=(.375, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J7',
        depth=-31
    )
    J7.buttonClock = core.Clock()
    J8 = visual.ButtonStim(win, 
        text='8', font='Arvo',
        pos=(.525, .075),
        letterHeight=0.05,
        size=(0.1, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(1.0000, 1.0000, -1.0000), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='J8',
        depth=-32
    )
    J8.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "escu" ---
    Pasdeffort = visual.ButtonStim(win, 
        text="Pas d'effort", font='Arvo',
        pos=(0, -0.4),
        letterHeight=0.03,
        size=(0.9, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Pasdeffort',
        depth=0
    )
    Pasdeffort.buttonClock = core.Clock()
    __1 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, -0.35),
        letterHeight=0.03,
        size=(0.85, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__1',
        depth=-1
    )
    __1.buttonClock = core.Clock()
    Trespeudeffort = visual.ButtonStim(win, 
        text="Très peu d'effort", font='Arvo',
        pos=(0, -0.3),
        letterHeight=0.03,
        size=(0.8, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Trespeudeffort',
        depth=-2
    )
    Trespeudeffort.buttonClock = core.Clock()
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "trial" ---
    # create an object to store info about Routine trial
    trial = data.Routine(
        name='trial',
        components=[text_instruction, R1, R2, R3, R4, R5, R6, R7, R8, B1, B2, B3, B4, B5, B6, B7, B8, V1, V2, V3, V4, V5, V6, V7, V8, J1, J2, J3, J4, J5, J6, J7, J8],
    )
    trial.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset R1 to account for continued clicks & clear times on/off
    R1.reset()
    # reset R2 to account for continued clicks & clear times on/off
    R2.reset()
    # reset R3 to account for continued clicks & clear times on/off
    R3.reset()
    # reset R4 to account for continued clicks & clear times on/off
    R4.reset()
    # reset R5 to account for continued clicks & clear times on/off
    R5.reset()
    # reset R6 to account for continued clicks & clear times on/off
    R6.reset()
    # reset R7 to account for continued clicks & clear times on/off
    R7.reset()
    # reset R8 to account for continued clicks & clear times on/off
    R8.reset()
    # reset B1 to account for continued clicks & clear times on/off
    B1.reset()
    # reset B2 to account for continued clicks & clear times on/off
    B2.reset()
    # reset B3 to account for continued clicks & clear times on/off
    B3.reset()
    # reset B4 to account for continued clicks & clear times on/off
    B4.reset()
    # reset B5 to account for continued clicks & clear times on/off
    B5.reset()
    # reset B6 to account for continued clicks & clear times on/off
    B6.reset()
    # reset B7 to account for continued clicks & clear times on/off
    B7.reset()
    # reset B8 to account for continued clicks & clear times on/off
    B8.reset()
    # reset V1 to account for continued clicks & clear times on/off
    V1.reset()
    # reset V2 to account for continued clicks & clear times on/off
    V2.reset()
    # reset V3 to account for continued clicks & clear times on/off
    V3.reset()
    # reset V4 to account for continued clicks & clear times on/off
    V4.reset()
    # reset V5 to account for continued clicks & clear times on/off
    V5.reset()
    # reset V6 to account for continued clicks & clear times on/off
    V6.reset()
    # reset V7 to account for continued clicks & clear times on/off
    V7.reset()
    # reset V8 to account for continued clicks & clear times on/off
    V8.reset()
    # reset J1 to account for continued clicks & clear times on/off
    J1.reset()
    # reset J2 to account for continued clicks & clear times on/off
    J2.reset()
    # reset J3 to account for continued clicks & clear times on/off
    J3.reset()
    # reset J4 to account for continued clicks & clear times on/off
    J4.reset()
    # reset J5 to account for continued clicks & clear times on/off
    J5.reset()
    # reset J6 to account for continued clicks & clear times on/off
    J6.reset()
    # reset J7 to account for continued clicks & clear times on/off
    J7.reset()
    # reset J8 to account for continued clicks & clear times on/off
    J8.reset()
    # store start times for trial
    trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    trial.tStart = globalClock.getTime(format='float')
    trial.status = STARTED
    thisExp.addData('trial.started', trial.tStart)
    trial.maxDuration = None
    # keep track of which components have finished
    trialComponents = trial.components
    for thisComponent in trial.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial" ---
    thisExp.currentRoutine = trial
    trial.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruction* updates
        
        # if text_instruction is starting this frame...
        if text_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction.frameNStart = frameN  # exact frame index
            text_instruction.tStart = t  # local t and not account for scr refresh
            text_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction.started')
            # update status
            text_instruction.status = STARTED
            text_instruction.setAutoDraw(True)
        
        # if text_instruction is active this frame...
        if text_instruction.status == STARTED:
            # update params
            pass
        
        # if text_instruction is stopping this frame...
        if text_instruction.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_instruction.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_instruction.tStop = t  # not accounting for scr refresh
                text_instruction.tStopRefresh = tThisFlipGlobal  # on global time
                text_instruction.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_instruction.stopped')
                # update status
                text_instruction.status = FINISHED
                text_instruction.setAutoDraw(False)
        # *R1* updates
        
        # if R1 is starting this frame...
        if R1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R1.frameNStart = frameN  # exact frame index
            R1.tStart = t  # local t and not account for scr refresh
            R1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R1.started')
            # update status
            R1.status = STARTED
            win.callOnFlip(R1.buttonClock.reset)
            R1.setAutoDraw(True)
        
        # if R1 is active this frame...
        if R1.status == STARTED:
            # update params
            pass
            # check whether R1 has been pressed
            if R1.isClicked:
                if not R1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R1.timesOn.append(R1.buttonClock.getTime())
                    R1.timesOff.append(R1.buttonClock.getTime())
                elif len(R1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R1.timesOff[-1] = R1.buttonClock.getTime()
                if not R1.wasClicked:
                    # end routine when R1 is clicked
                    continueRoutine = False
                if not R1.wasClicked:
                    # run callback code when R1 is clicked
                    pass
        # take note of whether R1 was clicked, so that next frame we know if clicks are new
        R1.wasClicked = R1.isClicked and R1.status == STARTED
        # *R2* updates
        
        # if R2 is starting this frame...
        if R2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R2.frameNStart = frameN  # exact frame index
            R2.tStart = t  # local t and not account for scr refresh
            R2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R2.started')
            # update status
            R2.status = STARTED
            win.callOnFlip(R2.buttonClock.reset)
            R2.setAutoDraw(True)
        
        # if R2 is active this frame...
        if R2.status == STARTED:
            # update params
            pass
            # check whether R2 has been pressed
            if R2.isClicked:
                if not R2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R2.timesOn.append(R2.buttonClock.getTime())
                    R2.timesOff.append(R2.buttonClock.getTime())
                elif len(R2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R2.timesOff[-1] = R2.buttonClock.getTime()
                if not R2.wasClicked:
                    # end routine when R2 is clicked
                    continueRoutine = False
                if not R2.wasClicked:
                    # run callback code when R2 is clicked
                    pass
        # take note of whether R2 was clicked, so that next frame we know if clicks are new
        R2.wasClicked = R2.isClicked and R2.status == STARTED
        # *R3* updates
        
        # if R3 is starting this frame...
        if R3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R3.frameNStart = frameN  # exact frame index
            R3.tStart = t  # local t and not account for scr refresh
            R3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R3.started')
            # update status
            R3.status = STARTED
            win.callOnFlip(R3.buttonClock.reset)
            R3.setAutoDraw(True)
        
        # if R3 is active this frame...
        if R3.status == STARTED:
            # update params
            pass
            # check whether R3 has been pressed
            if R3.isClicked:
                if not R3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R3.timesOn.append(R3.buttonClock.getTime())
                    R3.timesOff.append(R3.buttonClock.getTime())
                elif len(R3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R3.timesOff[-1] = R3.buttonClock.getTime()
                if not R3.wasClicked:
                    # end routine when R3 is clicked
                    continueRoutine = False
                if not R3.wasClicked:
                    # run callback code when R3 is clicked
                    pass
        # take note of whether R3 was clicked, so that next frame we know if clicks are new
        R3.wasClicked = R3.isClicked and R3.status == STARTED
        # *R4* updates
        
        # if R4 is starting this frame...
        if R4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R4.frameNStart = frameN  # exact frame index
            R4.tStart = t  # local t and not account for scr refresh
            R4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R4.started')
            # update status
            R4.status = STARTED
            win.callOnFlip(R4.buttonClock.reset)
            R4.setAutoDraw(True)
        
        # if R4 is active this frame...
        if R4.status == STARTED:
            # update params
            pass
            # check whether R4 has been pressed
            if R4.isClicked:
                if not R4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R4.timesOn.append(R4.buttonClock.getTime())
                    R4.timesOff.append(R4.buttonClock.getTime())
                elif len(R4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R4.timesOff[-1] = R4.buttonClock.getTime()
                if not R4.wasClicked:
                    # end routine when R4 is clicked
                    continueRoutine = False
                if not R4.wasClicked:
                    # run callback code when R4 is clicked
                    pass
        # take note of whether R4 was clicked, so that next frame we know if clicks are new
        R4.wasClicked = R4.isClicked and R4.status == STARTED
        # *R5* updates
        
        # if R5 is starting this frame...
        if R5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R5.frameNStart = frameN  # exact frame index
            R5.tStart = t  # local t and not account for scr refresh
            R5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R5.started')
            # update status
            R5.status = STARTED
            win.callOnFlip(R5.buttonClock.reset)
            R5.setAutoDraw(True)
        
        # if R5 is active this frame...
        if R5.status == STARTED:
            # update params
            pass
            # check whether R5 has been pressed
            if R5.isClicked:
                if not R5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R5.timesOn.append(R5.buttonClock.getTime())
                    R5.timesOff.append(R5.buttonClock.getTime())
                elif len(R5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R5.timesOff[-1] = R5.buttonClock.getTime()
                if not R5.wasClicked:
                    # end routine when R5 is clicked
                    continueRoutine = False
                if not R5.wasClicked:
                    # run callback code when R5 is clicked
                    pass
        # take note of whether R5 was clicked, so that next frame we know if clicks are new
        R5.wasClicked = R5.isClicked and R5.status == STARTED
        # *R6* updates
        
        # if R6 is starting this frame...
        if R6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R6.frameNStart = frameN  # exact frame index
            R6.tStart = t  # local t and not account for scr refresh
            R6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R6.started')
            # update status
            R6.status = STARTED
            win.callOnFlip(R6.buttonClock.reset)
            R6.setAutoDraw(True)
        
        # if R6 is active this frame...
        if R6.status == STARTED:
            # update params
            pass
            # check whether R6 has been pressed
            if R6.isClicked:
                if not R6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R6.timesOn.append(R6.buttonClock.getTime())
                    R6.timesOff.append(R6.buttonClock.getTime())
                elif len(R6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R6.timesOff[-1] = R6.buttonClock.getTime()
                if not R6.wasClicked:
                    # end routine when R6 is clicked
                    continueRoutine = False
                if not R6.wasClicked:
                    # run callback code when R6 is clicked
                    pass
        # take note of whether R6 was clicked, so that next frame we know if clicks are new
        R6.wasClicked = R6.isClicked and R6.status == STARTED
        # *R7* updates
        
        # if R7 is starting this frame...
        if R7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R7.frameNStart = frameN  # exact frame index
            R7.tStart = t  # local t and not account for scr refresh
            R7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R7.started')
            # update status
            R7.status = STARTED
            win.callOnFlip(R7.buttonClock.reset)
            R7.setAutoDraw(True)
        
        # if R7 is active this frame...
        if R7.status == STARTED:
            # update params
            pass
            # check whether R7 has been pressed
            if R7.isClicked:
                if not R7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R7.timesOn.append(R7.buttonClock.getTime())
                    R7.timesOff.append(R7.buttonClock.getTime())
                elif len(R7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R7.timesOff[-1] = R7.buttonClock.getTime()
                if not R7.wasClicked:
                    # end routine when R7 is clicked
                    continueRoutine = False
                if not R7.wasClicked:
                    # run callback code when R7 is clicked
                    pass
        # take note of whether R7 was clicked, so that next frame we know if clicks are new
        R7.wasClicked = R7.isClicked and R7.status == STARTED
        # *R8* updates
        
        # if R8 is starting this frame...
        if R8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R8.frameNStart = frameN  # exact frame index
            R8.tStart = t  # local t and not account for scr refresh
            R8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R8.started')
            # update status
            R8.status = STARTED
            win.callOnFlip(R8.buttonClock.reset)
            R8.setAutoDraw(True)
        
        # if R8 is active this frame...
        if R8.status == STARTED:
            # update params
            pass
            # check whether R8 has been pressed
            if R8.isClicked:
                if not R8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R8.timesOn.append(R8.buttonClock.getTime())
                    R8.timesOff.append(R8.buttonClock.getTime())
                elif len(R8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R8.timesOff[-1] = R8.buttonClock.getTime()
                if not R8.wasClicked:
                    # end routine when R8 is clicked
                    continueRoutine = False
                if not R8.wasClicked:
                    # run callback code when R8 is clicked
                    pass
        # take note of whether R8 was clicked, so that next frame we know if clicks are new
        R8.wasClicked = R8.isClicked and R8.status == STARTED
        # *B1* updates
        
        # if B1 is starting this frame...
        if B1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B1.frameNStart = frameN  # exact frame index
            B1.tStart = t  # local t and not account for scr refresh
            B1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B1.started')
            # update status
            B1.status = STARTED
            win.callOnFlip(B1.buttonClock.reset)
            B1.setAutoDraw(True)
        
        # if B1 is active this frame...
        if B1.status == STARTED:
            # update params
            pass
            # check whether B1 has been pressed
            if B1.isClicked:
                if not B1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B1.timesOn.append(B1.buttonClock.getTime())
                    B1.timesOff.append(B1.buttonClock.getTime())
                elif len(B1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B1.timesOff[-1] = B1.buttonClock.getTime()
                if not B1.wasClicked:
                    # end routine when B1 is clicked
                    continueRoutine = False
                if not B1.wasClicked:
                    # run callback code when B1 is clicked
                    pass
        # take note of whether B1 was clicked, so that next frame we know if clicks are new
        B1.wasClicked = B1.isClicked and B1.status == STARTED
        # *B2* updates
        
        # if B2 is starting this frame...
        if B2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B2.frameNStart = frameN  # exact frame index
            B2.tStart = t  # local t and not account for scr refresh
            B2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B2.started')
            # update status
            B2.status = STARTED
            win.callOnFlip(B2.buttonClock.reset)
            B2.setAutoDraw(True)
        
        # if B2 is active this frame...
        if B2.status == STARTED:
            # update params
            pass
            # check whether B2 has been pressed
            if B2.isClicked:
                if not B2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B2.timesOn.append(B2.buttonClock.getTime())
                    B2.timesOff.append(B2.buttonClock.getTime())
                elif len(B2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B2.timesOff[-1] = B2.buttonClock.getTime()
                if not B2.wasClicked:
                    # end routine when B2 is clicked
                    continueRoutine = False
                if not B2.wasClicked:
                    # run callback code when B2 is clicked
                    pass
        # take note of whether B2 was clicked, so that next frame we know if clicks are new
        B2.wasClicked = B2.isClicked and B2.status == STARTED
        # *B3* updates
        
        # if B3 is starting this frame...
        if B3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B3.frameNStart = frameN  # exact frame index
            B3.tStart = t  # local t and not account for scr refresh
            B3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B3.started')
            # update status
            B3.status = STARTED
            win.callOnFlip(B3.buttonClock.reset)
            B3.setAutoDraw(True)
        
        # if B3 is active this frame...
        if B3.status == STARTED:
            # update params
            pass
            # check whether B3 has been pressed
            if B3.isClicked:
                if not B3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B3.timesOn.append(B3.buttonClock.getTime())
                    B3.timesOff.append(B3.buttonClock.getTime())
                elif len(B3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B3.timesOff[-1] = B3.buttonClock.getTime()
                if not B3.wasClicked:
                    # end routine when B3 is clicked
                    continueRoutine = False
                if not B3.wasClicked:
                    # run callback code when B3 is clicked
                    pass
        # take note of whether B3 was clicked, so that next frame we know if clicks are new
        B3.wasClicked = B3.isClicked and B3.status == STARTED
        # *B4* updates
        
        # if B4 is starting this frame...
        if B4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B4.frameNStart = frameN  # exact frame index
            B4.tStart = t  # local t and not account for scr refresh
            B4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B4.started')
            # update status
            B4.status = STARTED
            win.callOnFlip(B4.buttonClock.reset)
            B4.setAutoDraw(True)
        
        # if B4 is active this frame...
        if B4.status == STARTED:
            # update params
            pass
            # check whether B4 has been pressed
            if B4.isClicked:
                if not B4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B4.timesOn.append(B4.buttonClock.getTime())
                    B4.timesOff.append(B4.buttonClock.getTime())
                elif len(B4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B4.timesOff[-1] = B4.buttonClock.getTime()
                if not B4.wasClicked:
                    # end routine when B4 is clicked
                    continueRoutine = False
                if not B4.wasClicked:
                    # run callback code when B4 is clicked
                    pass
        # take note of whether B4 was clicked, so that next frame we know if clicks are new
        B4.wasClicked = B4.isClicked and B4.status == STARTED
        # *B5* updates
        
        # if B5 is starting this frame...
        if B5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B5.frameNStart = frameN  # exact frame index
            B5.tStart = t  # local t and not account for scr refresh
            B5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B5.started')
            # update status
            B5.status = STARTED
            win.callOnFlip(B5.buttonClock.reset)
            B5.setAutoDraw(True)
        
        # if B5 is active this frame...
        if B5.status == STARTED:
            # update params
            pass
            # check whether B5 has been pressed
            if B5.isClicked:
                if not B5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B5.timesOn.append(B5.buttonClock.getTime())
                    B5.timesOff.append(B5.buttonClock.getTime())
                elif len(B5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B5.timesOff[-1] = B5.buttonClock.getTime()
                if not B5.wasClicked:
                    # end routine when B5 is clicked
                    continueRoutine = False
                if not B5.wasClicked:
                    # run callback code when B5 is clicked
                    pass
        # take note of whether B5 was clicked, so that next frame we know if clicks are new
        B5.wasClicked = B5.isClicked and B5.status == STARTED
        # *B6* updates
        
        # if B6 is starting this frame...
        if B6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B6.frameNStart = frameN  # exact frame index
            B6.tStart = t  # local t and not account for scr refresh
            B6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B6.started')
            # update status
            B6.status = STARTED
            win.callOnFlip(B6.buttonClock.reset)
            B6.setAutoDraw(True)
        
        # if B6 is active this frame...
        if B6.status == STARTED:
            # update params
            pass
            # check whether B6 has been pressed
            if B6.isClicked:
                if not B6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B6.timesOn.append(B6.buttonClock.getTime())
                    B6.timesOff.append(B6.buttonClock.getTime())
                elif len(B6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B6.timesOff[-1] = B6.buttonClock.getTime()
                if not B6.wasClicked:
                    # end routine when B6 is clicked
                    continueRoutine = False
                if not B6.wasClicked:
                    # run callback code when B6 is clicked
                    pass
        # take note of whether B6 was clicked, so that next frame we know if clicks are new
        B6.wasClicked = B6.isClicked and B6.status == STARTED
        # *B7* updates
        
        # if B7 is starting this frame...
        if B7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B7.frameNStart = frameN  # exact frame index
            B7.tStart = t  # local t and not account for scr refresh
            B7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B7.started')
            # update status
            B7.status = STARTED
            win.callOnFlip(B7.buttonClock.reset)
            B7.setAutoDraw(True)
        
        # if B7 is active this frame...
        if B7.status == STARTED:
            # update params
            pass
            # check whether B7 has been pressed
            if B7.isClicked:
                if not B7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B7.timesOn.append(B7.buttonClock.getTime())
                    B7.timesOff.append(B7.buttonClock.getTime())
                elif len(B7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B7.timesOff[-1] = B7.buttonClock.getTime()
                if not B7.wasClicked:
                    # end routine when B7 is clicked
                    continueRoutine = False
                if not B7.wasClicked:
                    # run callback code when B7 is clicked
                    pass
        # take note of whether B7 was clicked, so that next frame we know if clicks are new
        B7.wasClicked = B7.isClicked and B7.status == STARTED
        # *B8* updates
        
        # if B8 is starting this frame...
        if B8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B8.frameNStart = frameN  # exact frame index
            B8.tStart = t  # local t and not account for scr refresh
            B8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B8.started')
            # update status
            B8.status = STARTED
            win.callOnFlip(B8.buttonClock.reset)
            B8.setAutoDraw(True)
        
        # if B8 is active this frame...
        if B8.status == STARTED:
            # update params
            pass
            # check whether B8 has been pressed
            if B8.isClicked:
                if not B8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B8.timesOn.append(B8.buttonClock.getTime())
                    B8.timesOff.append(B8.buttonClock.getTime())
                elif len(B8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B8.timesOff[-1] = B8.buttonClock.getTime()
                if not B8.wasClicked:
                    # end routine when B8 is clicked
                    continueRoutine = False
                if not B8.wasClicked:
                    # run callback code when B8 is clicked
                    pass
        # take note of whether B8 was clicked, so that next frame we know if clicks are new
        B8.wasClicked = B8.isClicked and B8.status == STARTED
        # *V1* updates
        
        # if V1 is starting this frame...
        if V1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V1.frameNStart = frameN  # exact frame index
            V1.tStart = t  # local t and not account for scr refresh
            V1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V1.started')
            # update status
            V1.status = STARTED
            win.callOnFlip(V1.buttonClock.reset)
            V1.setAutoDraw(True)
        
        # if V1 is active this frame...
        if V1.status == STARTED:
            # update params
            pass
            # check whether V1 has been pressed
            if V1.isClicked:
                if not V1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V1.timesOn.append(V1.buttonClock.getTime())
                    V1.timesOff.append(V1.buttonClock.getTime())
                elif len(V1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V1.timesOff[-1] = V1.buttonClock.getTime()
                if not V1.wasClicked:
                    # end routine when V1 is clicked
                    continueRoutine = False
                if not V1.wasClicked:
                    # run callback code when V1 is clicked
                    pass
        # take note of whether V1 was clicked, so that next frame we know if clicks are new
        V1.wasClicked = V1.isClicked and V1.status == STARTED
        # *V2* updates
        
        # if V2 is starting this frame...
        if V2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V2.frameNStart = frameN  # exact frame index
            V2.tStart = t  # local t and not account for scr refresh
            V2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V2.started')
            # update status
            V2.status = STARTED
            win.callOnFlip(V2.buttonClock.reset)
            V2.setAutoDraw(True)
        
        # if V2 is active this frame...
        if V2.status == STARTED:
            # update params
            pass
            # check whether V2 has been pressed
            if V2.isClicked:
                if not V2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V2.timesOn.append(V2.buttonClock.getTime())
                    V2.timesOff.append(V2.buttonClock.getTime())
                elif len(V2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V2.timesOff[-1] = V2.buttonClock.getTime()
                if not V2.wasClicked:
                    # end routine when V2 is clicked
                    continueRoutine = False
                if not V2.wasClicked:
                    # run callback code when V2 is clicked
                    pass
        # take note of whether V2 was clicked, so that next frame we know if clicks are new
        V2.wasClicked = V2.isClicked and V2.status == STARTED
        # *V3* updates
        
        # if V3 is starting this frame...
        if V3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V3.frameNStart = frameN  # exact frame index
            V3.tStart = t  # local t and not account for scr refresh
            V3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V3.started')
            # update status
            V3.status = STARTED
            win.callOnFlip(V3.buttonClock.reset)
            V3.setAutoDraw(True)
        
        # if V3 is active this frame...
        if V3.status == STARTED:
            # update params
            pass
            # check whether V3 has been pressed
            if V3.isClicked:
                if not V3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V3.timesOn.append(V3.buttonClock.getTime())
                    V3.timesOff.append(V3.buttonClock.getTime())
                elif len(V3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V3.timesOff[-1] = V3.buttonClock.getTime()
                if not V3.wasClicked:
                    # end routine when V3 is clicked
                    continueRoutine = False
                if not V3.wasClicked:
                    # run callback code when V3 is clicked
                    pass
        # take note of whether V3 was clicked, so that next frame we know if clicks are new
        V3.wasClicked = V3.isClicked and V3.status == STARTED
        # *V4* updates
        
        # if V4 is starting this frame...
        if V4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V4.frameNStart = frameN  # exact frame index
            V4.tStart = t  # local t and not account for scr refresh
            V4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V4.started')
            # update status
            V4.status = STARTED
            win.callOnFlip(V4.buttonClock.reset)
            V4.setAutoDraw(True)
        
        # if V4 is active this frame...
        if V4.status == STARTED:
            # update params
            pass
            # check whether V4 has been pressed
            if V4.isClicked:
                if not V4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V4.timesOn.append(V4.buttonClock.getTime())
                    V4.timesOff.append(V4.buttonClock.getTime())
                elif len(V4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V4.timesOff[-1] = V4.buttonClock.getTime()
                if not V4.wasClicked:
                    # end routine when V4 is clicked
                    continueRoutine = False
                if not V4.wasClicked:
                    # run callback code when V4 is clicked
                    pass
        # take note of whether V4 was clicked, so that next frame we know if clicks are new
        V4.wasClicked = V4.isClicked and V4.status == STARTED
        # *V5* updates
        
        # if V5 is starting this frame...
        if V5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V5.frameNStart = frameN  # exact frame index
            V5.tStart = t  # local t and not account for scr refresh
            V5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V5.started')
            # update status
            V5.status = STARTED
            win.callOnFlip(V5.buttonClock.reset)
            V5.setAutoDraw(True)
        
        # if V5 is active this frame...
        if V5.status == STARTED:
            # update params
            pass
            # check whether V5 has been pressed
            if V5.isClicked:
                if not V5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V5.timesOn.append(V5.buttonClock.getTime())
                    V5.timesOff.append(V5.buttonClock.getTime())
                elif len(V5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V5.timesOff[-1] = V5.buttonClock.getTime()
                if not V5.wasClicked:
                    # end routine when V5 is clicked
                    continueRoutine = False
                if not V5.wasClicked:
                    # run callback code when V5 is clicked
                    pass
        # take note of whether V5 was clicked, so that next frame we know if clicks are new
        V5.wasClicked = V5.isClicked and V5.status == STARTED
        # *V6* updates
        
        # if V6 is starting this frame...
        if V6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V6.frameNStart = frameN  # exact frame index
            V6.tStart = t  # local t and not account for scr refresh
            V6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V6.started')
            # update status
            V6.status = STARTED
            win.callOnFlip(V6.buttonClock.reset)
            V6.setAutoDraw(True)
        
        # if V6 is active this frame...
        if V6.status == STARTED:
            # update params
            pass
            # check whether V6 has been pressed
            if V6.isClicked:
                if not V6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V6.timesOn.append(V6.buttonClock.getTime())
                    V6.timesOff.append(V6.buttonClock.getTime())
                elif len(V6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V6.timesOff[-1] = V6.buttonClock.getTime()
                if not V6.wasClicked:
                    # end routine when V6 is clicked
                    continueRoutine = False
                if not V6.wasClicked:
                    # run callback code when V6 is clicked
                    pass
        # take note of whether V6 was clicked, so that next frame we know if clicks are new
        V6.wasClicked = V6.isClicked and V6.status == STARTED
        # *V7* updates
        
        # if V7 is starting this frame...
        if V7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V7.frameNStart = frameN  # exact frame index
            V7.tStart = t  # local t and not account for scr refresh
            V7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V7.started')
            # update status
            V7.status = STARTED
            win.callOnFlip(V7.buttonClock.reset)
            V7.setAutoDraw(True)
        
        # if V7 is active this frame...
        if V7.status == STARTED:
            # update params
            pass
            # check whether V7 has been pressed
            if V7.isClicked:
                if not V7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V7.timesOn.append(V7.buttonClock.getTime())
                    V7.timesOff.append(V7.buttonClock.getTime())
                elif len(V7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V7.timesOff[-1] = V7.buttonClock.getTime()
                if not V7.wasClicked:
                    # end routine when V7 is clicked
                    continueRoutine = False
                if not V7.wasClicked:
                    # run callback code when V7 is clicked
                    pass
        # take note of whether V7 was clicked, so that next frame we know if clicks are new
        V7.wasClicked = V7.isClicked and V7.status == STARTED
        # *V8* updates
        
        # if V8 is starting this frame...
        if V8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V8.frameNStart = frameN  # exact frame index
            V8.tStart = t  # local t and not account for scr refresh
            V8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V8.started')
            # update status
            V8.status = STARTED
            win.callOnFlip(V8.buttonClock.reset)
            V8.setAutoDraw(True)
        
        # if V8 is active this frame...
        if V8.status == STARTED:
            # update params
            pass
            # check whether V8 has been pressed
            if V8.isClicked:
                if not V8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V8.timesOn.append(V8.buttonClock.getTime())
                    V8.timesOff.append(V8.buttonClock.getTime())
                elif len(V8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V8.timesOff[-1] = V8.buttonClock.getTime()
                if not V8.wasClicked:
                    # end routine when V8 is clicked
                    continueRoutine = False
                if not V8.wasClicked:
                    # run callback code when V8 is clicked
                    pass
        # take note of whether V8 was clicked, so that next frame we know if clicks are new
        V8.wasClicked = V8.isClicked and V8.status == STARTED
        # *J1* updates
        
        # if J1 is starting this frame...
        if J1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J1.frameNStart = frameN  # exact frame index
            J1.tStart = t  # local t and not account for scr refresh
            J1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J1.started')
            # update status
            J1.status = STARTED
            win.callOnFlip(J1.buttonClock.reset)
            J1.setAutoDraw(True)
        
        # if J1 is active this frame...
        if J1.status == STARTED:
            # update params
            pass
            # check whether J1 has been pressed
            if J1.isClicked:
                if not J1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J1.timesOn.append(J1.buttonClock.getTime())
                    J1.timesOff.append(J1.buttonClock.getTime())
                elif len(J1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J1.timesOff[-1] = J1.buttonClock.getTime()
                if not J1.wasClicked:
                    # end routine when J1 is clicked
                    continueRoutine = False
                if not J1.wasClicked:
                    # run callback code when J1 is clicked
                    pass
        # take note of whether J1 was clicked, so that next frame we know if clicks are new
        J1.wasClicked = J1.isClicked and J1.status == STARTED
        # *J2* updates
        
        # if J2 is starting this frame...
        if J2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J2.frameNStart = frameN  # exact frame index
            J2.tStart = t  # local t and not account for scr refresh
            J2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J2.started')
            # update status
            J2.status = STARTED
            win.callOnFlip(J2.buttonClock.reset)
            J2.setAutoDraw(True)
        
        # if J2 is active this frame...
        if J2.status == STARTED:
            # update params
            pass
            # check whether J2 has been pressed
            if J2.isClicked:
                if not J2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J2.timesOn.append(J2.buttonClock.getTime())
                    J2.timesOff.append(J2.buttonClock.getTime())
                elif len(J2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J2.timesOff[-1] = J2.buttonClock.getTime()
                if not J2.wasClicked:
                    # end routine when J2 is clicked
                    continueRoutine = False
                if not J2.wasClicked:
                    # run callback code when J2 is clicked
                    pass
        # take note of whether J2 was clicked, so that next frame we know if clicks are new
        J2.wasClicked = J2.isClicked and J2.status == STARTED
        # *J3* updates
        
        # if J3 is starting this frame...
        if J3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J3.frameNStart = frameN  # exact frame index
            J3.tStart = t  # local t and not account for scr refresh
            J3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J3.started')
            # update status
            J3.status = STARTED
            win.callOnFlip(J3.buttonClock.reset)
            J3.setAutoDraw(True)
        
        # if J3 is active this frame...
        if J3.status == STARTED:
            # update params
            pass
            # check whether J3 has been pressed
            if J3.isClicked:
                if not J3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J3.timesOn.append(J3.buttonClock.getTime())
                    J3.timesOff.append(J3.buttonClock.getTime())
                elif len(J3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J3.timesOff[-1] = J3.buttonClock.getTime()
                if not J3.wasClicked:
                    # end routine when J3 is clicked
                    continueRoutine = False
                if not J3.wasClicked:
                    # run callback code when J3 is clicked
                    pass
        # take note of whether J3 was clicked, so that next frame we know if clicks are new
        J3.wasClicked = J3.isClicked and J3.status == STARTED
        # *J4* updates
        
        # if J4 is starting this frame...
        if J4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J4.frameNStart = frameN  # exact frame index
            J4.tStart = t  # local t and not account for scr refresh
            J4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J4.started')
            # update status
            J4.status = STARTED
            win.callOnFlip(J4.buttonClock.reset)
            J4.setAutoDraw(True)
        
        # if J4 is active this frame...
        if J4.status == STARTED:
            # update params
            pass
            # check whether J4 has been pressed
            if J4.isClicked:
                if not J4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J4.timesOn.append(J4.buttonClock.getTime())
                    J4.timesOff.append(J4.buttonClock.getTime())
                elif len(J4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J4.timesOff[-1] = J4.buttonClock.getTime()
                if not J4.wasClicked:
                    # end routine when J4 is clicked
                    continueRoutine = False
                if not J4.wasClicked:
                    # run callback code when J4 is clicked
                    pass
        # take note of whether J4 was clicked, so that next frame we know if clicks are new
        J4.wasClicked = J4.isClicked and J4.status == STARTED
        # *J5* updates
        
        # if J5 is starting this frame...
        if J5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J5.frameNStart = frameN  # exact frame index
            J5.tStart = t  # local t and not account for scr refresh
            J5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J5.started')
            # update status
            J5.status = STARTED
            win.callOnFlip(J5.buttonClock.reset)
            J5.setAutoDraw(True)
        
        # if J5 is active this frame...
        if J5.status == STARTED:
            # update params
            pass
            # check whether J5 has been pressed
            if J5.isClicked:
                if not J5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J5.timesOn.append(J5.buttonClock.getTime())
                    J5.timesOff.append(J5.buttonClock.getTime())
                elif len(J5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J5.timesOff[-1] = J5.buttonClock.getTime()
                if not J5.wasClicked:
                    # end routine when J5 is clicked
                    continueRoutine = False
                if not J5.wasClicked:
                    # run callback code when J5 is clicked
                    pass
        # take note of whether J5 was clicked, so that next frame we know if clicks are new
        J5.wasClicked = J5.isClicked and J5.status == STARTED
        # *J6* updates
        
        # if J6 is starting this frame...
        if J6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J6.frameNStart = frameN  # exact frame index
            J6.tStart = t  # local t and not account for scr refresh
            J6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J6.started')
            # update status
            J6.status = STARTED
            win.callOnFlip(J6.buttonClock.reset)
            J6.setAutoDraw(True)
        
        # if J6 is active this frame...
        if J6.status == STARTED:
            # update params
            pass
            # check whether J6 has been pressed
            if J6.isClicked:
                if not J6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J6.timesOn.append(J6.buttonClock.getTime())
                    J6.timesOff.append(J6.buttonClock.getTime())
                elif len(J6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J6.timesOff[-1] = J6.buttonClock.getTime()
                if not J6.wasClicked:
                    # end routine when J6 is clicked
                    continueRoutine = False
                if not J6.wasClicked:
                    # run callback code when J6 is clicked
                    pass
        # take note of whether J6 was clicked, so that next frame we know if clicks are new
        J6.wasClicked = J6.isClicked and J6.status == STARTED
        # *J7* updates
        
        # if J7 is starting this frame...
        if J7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J7.frameNStart = frameN  # exact frame index
            J7.tStart = t  # local t and not account for scr refresh
            J7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J7.started')
            # update status
            J7.status = STARTED
            win.callOnFlip(J7.buttonClock.reset)
            J7.setAutoDraw(True)
        
        # if J7 is active this frame...
        if J7.status == STARTED:
            # update params
            pass
            # check whether J7 has been pressed
            if J7.isClicked:
                if not J7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J7.timesOn.append(J7.buttonClock.getTime())
                    J7.timesOff.append(J7.buttonClock.getTime())
                elif len(J7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J7.timesOff[-1] = J7.buttonClock.getTime()
                if not J7.wasClicked:
                    # end routine when J7 is clicked
                    continueRoutine = False
                if not J7.wasClicked:
                    # run callback code when J7 is clicked
                    pass
        # take note of whether J7 was clicked, so that next frame we know if clicks are new
        J7.wasClicked = J7.isClicked and J7.status == STARTED
        # *J8* updates
        
        # if J8 is starting this frame...
        if J8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J8.frameNStart = frameN  # exact frame index
            J8.tStart = t  # local t and not account for scr refresh
            J8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J8.started')
            # update status
            J8.status = STARTED
            win.callOnFlip(J8.buttonClock.reset)
            J8.setAutoDraw(True)
        
        # if J8 is active this frame...
        if J8.status == STARTED:
            # update params
            pass
            # check whether J8 has been pressed
            if J8.isClicked:
                if not J8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J8.timesOn.append(J8.buttonClock.getTime())
                    J8.timesOff.append(J8.buttonClock.getTime())
                elif len(J8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J8.timesOff[-1] = J8.buttonClock.getTime()
                if not J8.wasClicked:
                    # end routine when J8 is clicked
                    continueRoutine = False
                if not J8.wasClicked:
                    # run callback code when J8 is clicked
                    pass
        # take note of whether J8 was clicked, so that next frame we know if clicks are new
        J8.wasClicked = J8.isClicked and J8.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=trial,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            trial.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if trial.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in trial.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial" ---
    for thisComponent in trial.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for trial
    trial.tStop = globalClock.getTime(format='float')
    trial.tStopRefresh = tThisFlipGlobal
    thisExp.addData('trial.stopped', trial.tStop)
    thisExp.addData('R1.numClicks', R1.numClicks)
    if R1.numClicks:
       thisExp.addData('R1.timesOn', R1.timesOn)
       thisExp.addData('R1.timesOff', R1.timesOff)
    else:
       thisExp.addData('R1.timesOn', "")
       thisExp.addData('R1.timesOff', "")
    thisExp.addData('R2.numClicks', R2.numClicks)
    if R2.numClicks:
       thisExp.addData('R2.timesOn', R2.timesOn)
       thisExp.addData('R2.timesOff', R2.timesOff)
    else:
       thisExp.addData('R2.timesOn', "")
       thisExp.addData('R2.timesOff', "")
    thisExp.addData('R3.numClicks', R3.numClicks)
    if R3.numClicks:
       thisExp.addData('R3.timesOn', R3.timesOn)
       thisExp.addData('R3.timesOff', R3.timesOff)
    else:
       thisExp.addData('R3.timesOn', "")
       thisExp.addData('R3.timesOff', "")
    thisExp.addData('R4.numClicks', R4.numClicks)
    if R4.numClicks:
       thisExp.addData('R4.timesOn', R4.timesOn)
       thisExp.addData('R4.timesOff', R4.timesOff)
    else:
       thisExp.addData('R4.timesOn', "")
       thisExp.addData('R4.timesOff', "")
    thisExp.addData('R5.numClicks', R5.numClicks)
    if R5.numClicks:
       thisExp.addData('R5.timesOn', R5.timesOn)
       thisExp.addData('R5.timesOff', R5.timesOff)
    else:
       thisExp.addData('R5.timesOn', "")
       thisExp.addData('R5.timesOff', "")
    thisExp.addData('R6.numClicks', R6.numClicks)
    if R6.numClicks:
       thisExp.addData('R6.timesOn', R6.timesOn)
       thisExp.addData('R6.timesOff', R6.timesOff)
    else:
       thisExp.addData('R6.timesOn', "")
       thisExp.addData('R6.timesOff', "")
    thisExp.addData('R7.numClicks', R7.numClicks)
    if R7.numClicks:
       thisExp.addData('R7.timesOn', R7.timesOn)
       thisExp.addData('R7.timesOff', R7.timesOff)
    else:
       thisExp.addData('R7.timesOn', "")
       thisExp.addData('R7.timesOff', "")
    thisExp.addData('R8.numClicks', R8.numClicks)
    if R8.numClicks:
       thisExp.addData('R8.timesOn', R8.timesOn)
       thisExp.addData('R8.timesOff', R8.timesOff)
    else:
       thisExp.addData('R8.timesOn', "")
       thisExp.addData('R8.timesOff', "")
    thisExp.addData('B1.numClicks', B1.numClicks)
    if B1.numClicks:
       thisExp.addData('B1.timesOn', B1.timesOn)
       thisExp.addData('B1.timesOff', B1.timesOff)
    else:
       thisExp.addData('B1.timesOn', "")
       thisExp.addData('B1.timesOff', "")
    thisExp.addData('B2.numClicks', B2.numClicks)
    if B2.numClicks:
       thisExp.addData('B2.timesOn', B2.timesOn)
       thisExp.addData('B2.timesOff', B2.timesOff)
    else:
       thisExp.addData('B2.timesOn', "")
       thisExp.addData('B2.timesOff', "")
    thisExp.addData('B3.numClicks', B3.numClicks)
    if B3.numClicks:
       thisExp.addData('B3.timesOn', B3.timesOn)
       thisExp.addData('B3.timesOff', B3.timesOff)
    else:
       thisExp.addData('B3.timesOn', "")
       thisExp.addData('B3.timesOff', "")
    thisExp.addData('B4.numClicks', B4.numClicks)
    if B4.numClicks:
       thisExp.addData('B4.timesOn', B4.timesOn)
       thisExp.addData('B4.timesOff', B4.timesOff)
    else:
       thisExp.addData('B4.timesOn', "")
       thisExp.addData('B4.timesOff', "")
    thisExp.addData('B5.numClicks', B5.numClicks)
    if B5.numClicks:
       thisExp.addData('B5.timesOn', B5.timesOn)
       thisExp.addData('B5.timesOff', B5.timesOff)
    else:
       thisExp.addData('B5.timesOn', "")
       thisExp.addData('B5.timesOff', "")
    thisExp.addData('B6.numClicks', B6.numClicks)
    if B6.numClicks:
       thisExp.addData('B6.timesOn', B6.timesOn)
       thisExp.addData('B6.timesOff', B6.timesOff)
    else:
       thisExp.addData('B6.timesOn', "")
       thisExp.addData('B6.timesOff', "")
    thisExp.addData('B7.numClicks', B7.numClicks)
    if B7.numClicks:
       thisExp.addData('B7.timesOn', B7.timesOn)
       thisExp.addData('B7.timesOff', B7.timesOff)
    else:
       thisExp.addData('B7.timesOn', "")
       thisExp.addData('B7.timesOff', "")
    thisExp.addData('B8.numClicks', B8.numClicks)
    if B8.numClicks:
       thisExp.addData('B8.timesOn', B8.timesOn)
       thisExp.addData('B8.timesOff', B8.timesOff)
    else:
       thisExp.addData('B8.timesOn', "")
       thisExp.addData('B8.timesOff', "")
    thisExp.addData('V1.numClicks', V1.numClicks)
    if V1.numClicks:
       thisExp.addData('V1.timesOn', V1.timesOn)
       thisExp.addData('V1.timesOff', V1.timesOff)
    else:
       thisExp.addData('V1.timesOn', "")
       thisExp.addData('V1.timesOff', "")
    thisExp.addData('V2.numClicks', V2.numClicks)
    if V2.numClicks:
       thisExp.addData('V2.timesOn', V2.timesOn)
       thisExp.addData('V2.timesOff', V2.timesOff)
    else:
       thisExp.addData('V2.timesOn', "")
       thisExp.addData('V2.timesOff', "")
    thisExp.addData('V3.numClicks', V3.numClicks)
    if V3.numClicks:
       thisExp.addData('V3.timesOn', V3.timesOn)
       thisExp.addData('V3.timesOff', V3.timesOff)
    else:
       thisExp.addData('V3.timesOn', "")
       thisExp.addData('V3.timesOff', "")
    thisExp.addData('V4.numClicks', V4.numClicks)
    if V4.numClicks:
       thisExp.addData('V4.timesOn', V4.timesOn)
       thisExp.addData('V4.timesOff', V4.timesOff)
    else:
       thisExp.addData('V4.timesOn', "")
       thisExp.addData('V4.timesOff', "")
    thisExp.addData('V5.numClicks', V5.numClicks)
    if V5.numClicks:
       thisExp.addData('V5.timesOn', V5.timesOn)
       thisExp.addData('V5.timesOff', V5.timesOff)
    else:
       thisExp.addData('V5.timesOn', "")
       thisExp.addData('V5.timesOff', "")
    thisExp.addData('V6.numClicks', V6.numClicks)
    if V6.numClicks:
       thisExp.addData('V6.timesOn', V6.timesOn)
       thisExp.addData('V6.timesOff', V6.timesOff)
    else:
       thisExp.addData('V6.timesOn', "")
       thisExp.addData('V6.timesOff', "")
    thisExp.addData('V7.numClicks', V7.numClicks)
    if V7.numClicks:
       thisExp.addData('V7.timesOn', V7.timesOn)
       thisExp.addData('V7.timesOff', V7.timesOff)
    else:
       thisExp.addData('V7.timesOn', "")
       thisExp.addData('V7.timesOff', "")
    thisExp.addData('V8.numClicks', V8.numClicks)
    if V8.numClicks:
       thisExp.addData('V8.timesOn', V8.timesOn)
       thisExp.addData('V8.timesOff', V8.timesOff)
    else:
       thisExp.addData('V8.timesOn', "")
       thisExp.addData('V8.timesOff', "")
    thisExp.addData('J1.numClicks', J1.numClicks)
    if J1.numClicks:
       thisExp.addData('J1.timesOn', J1.timesOn)
       thisExp.addData('J1.timesOff', J1.timesOff)
    else:
       thisExp.addData('J1.timesOn', "")
       thisExp.addData('J1.timesOff', "")
    thisExp.addData('J2.numClicks', J2.numClicks)
    if J2.numClicks:
       thisExp.addData('J2.timesOn', J2.timesOn)
       thisExp.addData('J2.timesOff', J2.timesOff)
    else:
       thisExp.addData('J2.timesOn', "")
       thisExp.addData('J2.timesOff', "")
    thisExp.addData('J3.numClicks', J3.numClicks)
    if J3.numClicks:
       thisExp.addData('J3.timesOn', J3.timesOn)
       thisExp.addData('J3.timesOff', J3.timesOff)
    else:
       thisExp.addData('J3.timesOn', "")
       thisExp.addData('J3.timesOff', "")
    thisExp.addData('J4.numClicks', J4.numClicks)
    if J4.numClicks:
       thisExp.addData('J4.timesOn', J4.timesOn)
       thisExp.addData('J4.timesOff', J4.timesOff)
    else:
       thisExp.addData('J4.timesOn', "")
       thisExp.addData('J4.timesOff', "")
    thisExp.addData('J5.numClicks', J5.numClicks)
    if J5.numClicks:
       thisExp.addData('J5.timesOn', J5.timesOn)
       thisExp.addData('J5.timesOff', J5.timesOff)
    else:
       thisExp.addData('J5.timesOn', "")
       thisExp.addData('J5.timesOff', "")
    thisExp.addData('J6.numClicks', J6.numClicks)
    if J6.numClicks:
       thisExp.addData('J6.timesOn', J6.timesOn)
       thisExp.addData('J6.timesOff', J6.timesOff)
    else:
       thisExp.addData('J6.timesOn', "")
       thisExp.addData('J6.timesOff', "")
    thisExp.addData('J7.numClicks', J7.numClicks)
    if J7.numClicks:
       thisExp.addData('J7.timesOn', J7.timesOn)
       thisExp.addData('J7.timesOff', J7.timesOff)
    else:
       thisExp.addData('J7.timesOn', "")
       thisExp.addData('J7.timesOff', "")
    thisExp.addData('J8.numClicks', J8.numClicks)
    if J8.numClicks:
       thisExp.addData('J8.timesOn', J8.timesOn)
       thisExp.addData('J8.timesOff', J8.timesOff)
    else:
       thisExp.addData('J8.timesOn', "")
       thisExp.addData('J8.timesOff', "")
    thisExp.nextEntry()
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "escu" ---
    # create an object to store info about Routine escu
    escu = data.Routine(
        name='escu',
        components=[Pasdeffort, __1, Trespeudeffort],
    )
    escu.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset Pasdeffort to account for continued clicks & clear times on/off
    Pasdeffort.reset()
    # reset __1 to account for continued clicks & clear times on/off
    __1.reset()
    # reset Trespeudeffort to account for continued clicks & clear times on/off
    Trespeudeffort.reset()
    # store start times for escu
    escu.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    escu.tStart = globalClock.getTime(format='float')
    escu.status = STARTED
    thisExp.addData('escu.started', escu.tStart)
    escu.maxDuration = None
    # keep track of which components have finished
    escuComponents = escu.components
    for thisComponent in escu.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "escu" ---
    thisExp.currentRoutine = escu
    escu.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *Pasdeffort* updates
        
        # if Pasdeffort is starting this frame...
        if Pasdeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Pasdeffort.frameNStart = frameN  # exact frame index
            Pasdeffort.tStart = t  # local t and not account for scr refresh
            Pasdeffort.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Pasdeffort, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Pasdeffort.started')
            # update status
            Pasdeffort.status = STARTED
            win.callOnFlip(Pasdeffort.buttonClock.reset)
            Pasdeffort.setAutoDraw(True)
        
        # if Pasdeffort is active this frame...
        if Pasdeffort.status == STARTED:
            # update params
            pass
            # check whether Pasdeffort has been pressed
            if Pasdeffort.isClicked:
                if not Pasdeffort.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    Pasdeffort.timesOn.append(Pasdeffort.buttonClock.getTime())
                    Pasdeffort.timesOff.append(Pasdeffort.buttonClock.getTime())
                elif len(Pasdeffort.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    Pasdeffort.timesOff[-1] = Pasdeffort.buttonClock.getTime()
                if not Pasdeffort.wasClicked:
                    # end routine when Pasdeffort is clicked
                    continueRoutine = False
                if not Pasdeffort.wasClicked:
                    # run callback code when Pasdeffort is clicked
                    pass
        # take note of whether Pasdeffort was clicked, so that next frame we know if clicks are new
        Pasdeffort.wasClicked = Pasdeffort.isClicked and Pasdeffort.status == STARTED
        # *__1* updates
        
        # if __1 is starting this frame...
        if __1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            __1.frameNStart = frameN  # exact frame index
            __1.tStart = t  # local t and not account for scr refresh
            __1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(__1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, '__1.started')
            # update status
            __1.status = STARTED
            win.callOnFlip(__1.buttonClock.reset)
            __1.setAutoDraw(True)
        
        # if __1 is active this frame...
        if __1.status == STARTED:
            # update params
            pass
            # check whether __1 has been pressed
            if __1.isClicked:
                if not __1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    __1.timesOn.append(__1.buttonClock.getTime())
                    __1.timesOff.append(__1.buttonClock.getTime())
                elif len(__1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    __1.timesOff[-1] = __1.buttonClock.getTime()
                if not __1.wasClicked:
                    # end routine when __1 is clicked
                    continueRoutine = False
                if not __1.wasClicked:
                    # run callback code when __1 is clicked
                    pass
        # take note of whether __1 was clicked, so that next frame we know if clicks are new
        __1.wasClicked = __1.isClicked and __1.status == STARTED
        # *Trespeudeffort* updates
        
        # if Trespeudeffort is starting this frame...
        if Trespeudeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Trespeudeffort.frameNStart = frameN  # exact frame index
            Trespeudeffort.tStart = t  # local t and not account for scr refresh
            Trespeudeffort.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Trespeudeffort, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Trespeudeffort.started')
            # update status
            Trespeudeffort.status = STARTED
            win.callOnFlip(Trespeudeffort.buttonClock.reset)
            Trespeudeffort.setAutoDraw(True)
        
        # if Trespeudeffort is active this frame...
        if Trespeudeffort.status == STARTED:
            # update params
            pass
            # check whether Trespeudeffort has been pressed
            if Trespeudeffort.isClicked:
                if not Trespeudeffort.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    Trespeudeffort.timesOn.append(Trespeudeffort.buttonClock.getTime())
                    Trespeudeffort.timesOff.append(Trespeudeffort.buttonClock.getTime())
                elif len(Trespeudeffort.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    Trespeudeffort.timesOff[-1] = Trespeudeffort.buttonClock.getTime()
                if not Trespeudeffort.wasClicked:
                    # end routine when Trespeudeffort is clicked
                    continueRoutine = False
                if not Trespeudeffort.wasClicked:
                    # run callback code when Trespeudeffort is clicked
                    pass
        # take note of whether Trespeudeffort was clicked, so that next frame we know if clicks are new
        Trespeudeffort.wasClicked = Trespeudeffort.isClicked and Trespeudeffort.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=escu,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            escu.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if escu.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in escu.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "escu" ---
    for thisComponent in escu.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for escu
    escu.tStop = globalClock.getTime(format='float')
    escu.tStopRefresh = tThisFlipGlobal
    thisExp.addData('escu.stopped', escu.tStop)
    thisExp.addData('Pasdeffort.numClicks', Pasdeffort.numClicks)
    if Pasdeffort.numClicks:
       thisExp.addData('Pasdeffort.timesOn', Pasdeffort.timesOn)
       thisExp.addData('Pasdeffort.timesOff', Pasdeffort.timesOff)
    else:
       thisExp.addData('Pasdeffort.timesOn', "")
       thisExp.addData('Pasdeffort.timesOff', "")
    thisExp.addData('__1.numClicks', __1.numClicks)
    if __1.numClicks:
       thisExp.addData('__1.timesOn', __1.timesOn)
       thisExp.addData('__1.timesOff', __1.timesOff)
    else:
       thisExp.addData('__1.timesOn', "")
       thisExp.addData('__1.timesOff', "")
    thisExp.addData('Trespeudeffort.numClicks', Trespeudeffort.numClicks)
    if Trespeudeffort.numClicks:
       thisExp.addData('Trespeudeffort.timesOn', Trespeudeffort.timesOn)
       thisExp.addData('Trespeudeffort.timesOff', Trespeudeffort.timesOff)
    else:
       thisExp.addData('Trespeudeffort.timesOn', "")
       thisExp.addData('Trespeudeffort.timesOff', "")
    thisExp.nextEntry()
    # the Routine "escu" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "trial" ---
    # create an object to store info about Routine trial
    trial = data.Routine(
        name='trial',
        components=[text_instruction, R1, R2, R3, R4, R5, R6, R7, R8, B1, B2, B3, B4, B5, B6, B7, B8, V1, V2, V3, V4, V5, V6, V7, V8, J1, J2, J3, J4, J5, J6, J7, J8],
    )
    trial.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset R1 to account for continued clicks & clear times on/off
    R1.reset()
    # reset R2 to account for continued clicks & clear times on/off
    R2.reset()
    # reset R3 to account for continued clicks & clear times on/off
    R3.reset()
    # reset R4 to account for continued clicks & clear times on/off
    R4.reset()
    # reset R5 to account for continued clicks & clear times on/off
    R5.reset()
    # reset R6 to account for continued clicks & clear times on/off
    R6.reset()
    # reset R7 to account for continued clicks & clear times on/off
    R7.reset()
    # reset R8 to account for continued clicks & clear times on/off
    R8.reset()
    # reset B1 to account for continued clicks & clear times on/off
    B1.reset()
    # reset B2 to account for continued clicks & clear times on/off
    B2.reset()
    # reset B3 to account for continued clicks & clear times on/off
    B3.reset()
    # reset B4 to account for continued clicks & clear times on/off
    B4.reset()
    # reset B5 to account for continued clicks & clear times on/off
    B5.reset()
    # reset B6 to account for continued clicks & clear times on/off
    B6.reset()
    # reset B7 to account for continued clicks & clear times on/off
    B7.reset()
    # reset B8 to account for continued clicks & clear times on/off
    B8.reset()
    # reset V1 to account for continued clicks & clear times on/off
    V1.reset()
    # reset V2 to account for continued clicks & clear times on/off
    V2.reset()
    # reset V3 to account for continued clicks & clear times on/off
    V3.reset()
    # reset V4 to account for continued clicks & clear times on/off
    V4.reset()
    # reset V5 to account for continued clicks & clear times on/off
    V5.reset()
    # reset V6 to account for continued clicks & clear times on/off
    V6.reset()
    # reset V7 to account for continued clicks & clear times on/off
    V7.reset()
    # reset V8 to account for continued clicks & clear times on/off
    V8.reset()
    # reset J1 to account for continued clicks & clear times on/off
    J1.reset()
    # reset J2 to account for continued clicks & clear times on/off
    J2.reset()
    # reset J3 to account for continued clicks & clear times on/off
    J3.reset()
    # reset J4 to account for continued clicks & clear times on/off
    J4.reset()
    # reset J5 to account for continued clicks & clear times on/off
    J5.reset()
    # reset J6 to account for continued clicks & clear times on/off
    J6.reset()
    # reset J7 to account for continued clicks & clear times on/off
    J7.reset()
    # reset J8 to account for continued clicks & clear times on/off
    J8.reset()
    # store start times for trial
    trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    trial.tStart = globalClock.getTime(format='float')
    trial.status = STARTED
    thisExp.addData('trial.started', trial.tStart)
    trial.maxDuration = None
    # keep track of which components have finished
    trialComponents = trial.components
    for thisComponent in trial.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial" ---
    thisExp.currentRoutine = trial
    trial.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruction* updates
        
        # if text_instruction is starting this frame...
        if text_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction.frameNStart = frameN  # exact frame index
            text_instruction.tStart = t  # local t and not account for scr refresh
            text_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction.started')
            # update status
            text_instruction.status = STARTED
            text_instruction.setAutoDraw(True)
        
        # if text_instruction is active this frame...
        if text_instruction.status == STARTED:
            # update params
            pass
        
        # if text_instruction is stopping this frame...
        if text_instruction.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_instruction.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_instruction.tStop = t  # not accounting for scr refresh
                text_instruction.tStopRefresh = tThisFlipGlobal  # on global time
                text_instruction.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_instruction.stopped')
                # update status
                text_instruction.status = FINISHED
                text_instruction.setAutoDraw(False)
        # *R1* updates
        
        # if R1 is starting this frame...
        if R1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R1.frameNStart = frameN  # exact frame index
            R1.tStart = t  # local t and not account for scr refresh
            R1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R1.started')
            # update status
            R1.status = STARTED
            win.callOnFlip(R1.buttonClock.reset)
            R1.setAutoDraw(True)
        
        # if R1 is active this frame...
        if R1.status == STARTED:
            # update params
            pass
            # check whether R1 has been pressed
            if R1.isClicked:
                if not R1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R1.timesOn.append(R1.buttonClock.getTime())
                    R1.timesOff.append(R1.buttonClock.getTime())
                elif len(R1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R1.timesOff[-1] = R1.buttonClock.getTime()
                if not R1.wasClicked:
                    # end routine when R1 is clicked
                    continueRoutine = False
                if not R1.wasClicked:
                    # run callback code when R1 is clicked
                    pass
        # take note of whether R1 was clicked, so that next frame we know if clicks are new
        R1.wasClicked = R1.isClicked and R1.status == STARTED
        # *R2* updates
        
        # if R2 is starting this frame...
        if R2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R2.frameNStart = frameN  # exact frame index
            R2.tStart = t  # local t and not account for scr refresh
            R2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R2.started')
            # update status
            R2.status = STARTED
            win.callOnFlip(R2.buttonClock.reset)
            R2.setAutoDraw(True)
        
        # if R2 is active this frame...
        if R2.status == STARTED:
            # update params
            pass
            # check whether R2 has been pressed
            if R2.isClicked:
                if not R2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R2.timesOn.append(R2.buttonClock.getTime())
                    R2.timesOff.append(R2.buttonClock.getTime())
                elif len(R2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R2.timesOff[-1] = R2.buttonClock.getTime()
                if not R2.wasClicked:
                    # end routine when R2 is clicked
                    continueRoutine = False
                if not R2.wasClicked:
                    # run callback code when R2 is clicked
                    pass
        # take note of whether R2 was clicked, so that next frame we know if clicks are new
        R2.wasClicked = R2.isClicked and R2.status == STARTED
        # *R3* updates
        
        # if R3 is starting this frame...
        if R3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R3.frameNStart = frameN  # exact frame index
            R3.tStart = t  # local t and not account for scr refresh
            R3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R3.started')
            # update status
            R3.status = STARTED
            win.callOnFlip(R3.buttonClock.reset)
            R3.setAutoDraw(True)
        
        # if R3 is active this frame...
        if R3.status == STARTED:
            # update params
            pass
            # check whether R3 has been pressed
            if R3.isClicked:
                if not R3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R3.timesOn.append(R3.buttonClock.getTime())
                    R3.timesOff.append(R3.buttonClock.getTime())
                elif len(R3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R3.timesOff[-1] = R3.buttonClock.getTime()
                if not R3.wasClicked:
                    # end routine when R3 is clicked
                    continueRoutine = False
                if not R3.wasClicked:
                    # run callback code when R3 is clicked
                    pass
        # take note of whether R3 was clicked, so that next frame we know if clicks are new
        R3.wasClicked = R3.isClicked and R3.status == STARTED
        # *R4* updates
        
        # if R4 is starting this frame...
        if R4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R4.frameNStart = frameN  # exact frame index
            R4.tStart = t  # local t and not account for scr refresh
            R4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R4.started')
            # update status
            R4.status = STARTED
            win.callOnFlip(R4.buttonClock.reset)
            R4.setAutoDraw(True)
        
        # if R4 is active this frame...
        if R4.status == STARTED:
            # update params
            pass
            # check whether R4 has been pressed
            if R4.isClicked:
                if not R4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R4.timesOn.append(R4.buttonClock.getTime())
                    R4.timesOff.append(R4.buttonClock.getTime())
                elif len(R4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R4.timesOff[-1] = R4.buttonClock.getTime()
                if not R4.wasClicked:
                    # end routine when R4 is clicked
                    continueRoutine = False
                if not R4.wasClicked:
                    # run callback code when R4 is clicked
                    pass
        # take note of whether R4 was clicked, so that next frame we know if clicks are new
        R4.wasClicked = R4.isClicked and R4.status == STARTED
        # *R5* updates
        
        # if R5 is starting this frame...
        if R5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R5.frameNStart = frameN  # exact frame index
            R5.tStart = t  # local t and not account for scr refresh
            R5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R5.started')
            # update status
            R5.status = STARTED
            win.callOnFlip(R5.buttonClock.reset)
            R5.setAutoDraw(True)
        
        # if R5 is active this frame...
        if R5.status == STARTED:
            # update params
            pass
            # check whether R5 has been pressed
            if R5.isClicked:
                if not R5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R5.timesOn.append(R5.buttonClock.getTime())
                    R5.timesOff.append(R5.buttonClock.getTime())
                elif len(R5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R5.timesOff[-1] = R5.buttonClock.getTime()
                if not R5.wasClicked:
                    # end routine when R5 is clicked
                    continueRoutine = False
                if not R5.wasClicked:
                    # run callback code when R5 is clicked
                    pass
        # take note of whether R5 was clicked, so that next frame we know if clicks are new
        R5.wasClicked = R5.isClicked and R5.status == STARTED
        # *R6* updates
        
        # if R6 is starting this frame...
        if R6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R6.frameNStart = frameN  # exact frame index
            R6.tStart = t  # local t and not account for scr refresh
            R6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R6.started')
            # update status
            R6.status = STARTED
            win.callOnFlip(R6.buttonClock.reset)
            R6.setAutoDraw(True)
        
        # if R6 is active this frame...
        if R6.status == STARTED:
            # update params
            pass
            # check whether R6 has been pressed
            if R6.isClicked:
                if not R6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R6.timesOn.append(R6.buttonClock.getTime())
                    R6.timesOff.append(R6.buttonClock.getTime())
                elif len(R6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R6.timesOff[-1] = R6.buttonClock.getTime()
                if not R6.wasClicked:
                    # end routine when R6 is clicked
                    continueRoutine = False
                if not R6.wasClicked:
                    # run callback code when R6 is clicked
                    pass
        # take note of whether R6 was clicked, so that next frame we know if clicks are new
        R6.wasClicked = R6.isClicked and R6.status == STARTED
        # *R7* updates
        
        # if R7 is starting this frame...
        if R7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R7.frameNStart = frameN  # exact frame index
            R7.tStart = t  # local t and not account for scr refresh
            R7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R7.started')
            # update status
            R7.status = STARTED
            win.callOnFlip(R7.buttonClock.reset)
            R7.setAutoDraw(True)
        
        # if R7 is active this frame...
        if R7.status == STARTED:
            # update params
            pass
            # check whether R7 has been pressed
            if R7.isClicked:
                if not R7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R7.timesOn.append(R7.buttonClock.getTime())
                    R7.timesOff.append(R7.buttonClock.getTime())
                elif len(R7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R7.timesOff[-1] = R7.buttonClock.getTime()
                if not R7.wasClicked:
                    # end routine when R7 is clicked
                    continueRoutine = False
                if not R7.wasClicked:
                    # run callback code when R7 is clicked
                    pass
        # take note of whether R7 was clicked, so that next frame we know if clicks are new
        R7.wasClicked = R7.isClicked and R7.status == STARTED
        # *R8* updates
        
        # if R8 is starting this frame...
        if R8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R8.frameNStart = frameN  # exact frame index
            R8.tStart = t  # local t and not account for scr refresh
            R8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R8.started')
            # update status
            R8.status = STARTED
            win.callOnFlip(R8.buttonClock.reset)
            R8.setAutoDraw(True)
        
        # if R8 is active this frame...
        if R8.status == STARTED:
            # update params
            pass
            # check whether R8 has been pressed
            if R8.isClicked:
                if not R8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R8.timesOn.append(R8.buttonClock.getTime())
                    R8.timesOff.append(R8.buttonClock.getTime())
                elif len(R8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R8.timesOff[-1] = R8.buttonClock.getTime()
                if not R8.wasClicked:
                    # end routine when R8 is clicked
                    continueRoutine = False
                if not R8.wasClicked:
                    # run callback code when R8 is clicked
                    pass
        # take note of whether R8 was clicked, so that next frame we know if clicks are new
        R8.wasClicked = R8.isClicked and R8.status == STARTED
        # *B1* updates
        
        # if B1 is starting this frame...
        if B1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B1.frameNStart = frameN  # exact frame index
            B1.tStart = t  # local t and not account for scr refresh
            B1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B1.started')
            # update status
            B1.status = STARTED
            win.callOnFlip(B1.buttonClock.reset)
            B1.setAutoDraw(True)
        
        # if B1 is active this frame...
        if B1.status == STARTED:
            # update params
            pass
            # check whether B1 has been pressed
            if B1.isClicked:
                if not B1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B1.timesOn.append(B1.buttonClock.getTime())
                    B1.timesOff.append(B1.buttonClock.getTime())
                elif len(B1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B1.timesOff[-1] = B1.buttonClock.getTime()
                if not B1.wasClicked:
                    # end routine when B1 is clicked
                    continueRoutine = False
                if not B1.wasClicked:
                    # run callback code when B1 is clicked
                    pass
        # take note of whether B1 was clicked, so that next frame we know if clicks are new
        B1.wasClicked = B1.isClicked and B1.status == STARTED
        # *B2* updates
        
        # if B2 is starting this frame...
        if B2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B2.frameNStart = frameN  # exact frame index
            B2.tStart = t  # local t and not account for scr refresh
            B2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B2.started')
            # update status
            B2.status = STARTED
            win.callOnFlip(B2.buttonClock.reset)
            B2.setAutoDraw(True)
        
        # if B2 is active this frame...
        if B2.status == STARTED:
            # update params
            pass
            # check whether B2 has been pressed
            if B2.isClicked:
                if not B2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B2.timesOn.append(B2.buttonClock.getTime())
                    B2.timesOff.append(B2.buttonClock.getTime())
                elif len(B2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B2.timesOff[-1] = B2.buttonClock.getTime()
                if not B2.wasClicked:
                    # end routine when B2 is clicked
                    continueRoutine = False
                if not B2.wasClicked:
                    # run callback code when B2 is clicked
                    pass
        # take note of whether B2 was clicked, so that next frame we know if clicks are new
        B2.wasClicked = B2.isClicked and B2.status == STARTED
        # *B3* updates
        
        # if B3 is starting this frame...
        if B3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B3.frameNStart = frameN  # exact frame index
            B3.tStart = t  # local t and not account for scr refresh
            B3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B3.started')
            # update status
            B3.status = STARTED
            win.callOnFlip(B3.buttonClock.reset)
            B3.setAutoDraw(True)
        
        # if B3 is active this frame...
        if B3.status == STARTED:
            # update params
            pass
            # check whether B3 has been pressed
            if B3.isClicked:
                if not B3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B3.timesOn.append(B3.buttonClock.getTime())
                    B3.timesOff.append(B3.buttonClock.getTime())
                elif len(B3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B3.timesOff[-1] = B3.buttonClock.getTime()
                if not B3.wasClicked:
                    # end routine when B3 is clicked
                    continueRoutine = False
                if not B3.wasClicked:
                    # run callback code when B3 is clicked
                    pass
        # take note of whether B3 was clicked, so that next frame we know if clicks are new
        B3.wasClicked = B3.isClicked and B3.status == STARTED
        # *B4* updates
        
        # if B4 is starting this frame...
        if B4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B4.frameNStart = frameN  # exact frame index
            B4.tStart = t  # local t and not account for scr refresh
            B4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B4.started')
            # update status
            B4.status = STARTED
            win.callOnFlip(B4.buttonClock.reset)
            B4.setAutoDraw(True)
        
        # if B4 is active this frame...
        if B4.status == STARTED:
            # update params
            pass
            # check whether B4 has been pressed
            if B4.isClicked:
                if not B4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B4.timesOn.append(B4.buttonClock.getTime())
                    B4.timesOff.append(B4.buttonClock.getTime())
                elif len(B4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B4.timesOff[-1] = B4.buttonClock.getTime()
                if not B4.wasClicked:
                    # end routine when B4 is clicked
                    continueRoutine = False
                if not B4.wasClicked:
                    # run callback code when B4 is clicked
                    pass
        # take note of whether B4 was clicked, so that next frame we know if clicks are new
        B4.wasClicked = B4.isClicked and B4.status == STARTED
        # *B5* updates
        
        # if B5 is starting this frame...
        if B5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B5.frameNStart = frameN  # exact frame index
            B5.tStart = t  # local t and not account for scr refresh
            B5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B5.started')
            # update status
            B5.status = STARTED
            win.callOnFlip(B5.buttonClock.reset)
            B5.setAutoDraw(True)
        
        # if B5 is active this frame...
        if B5.status == STARTED:
            # update params
            pass
            # check whether B5 has been pressed
            if B5.isClicked:
                if not B5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B5.timesOn.append(B5.buttonClock.getTime())
                    B5.timesOff.append(B5.buttonClock.getTime())
                elif len(B5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B5.timesOff[-1] = B5.buttonClock.getTime()
                if not B5.wasClicked:
                    # end routine when B5 is clicked
                    continueRoutine = False
                if not B5.wasClicked:
                    # run callback code when B5 is clicked
                    pass
        # take note of whether B5 was clicked, so that next frame we know if clicks are new
        B5.wasClicked = B5.isClicked and B5.status == STARTED
        # *B6* updates
        
        # if B6 is starting this frame...
        if B6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B6.frameNStart = frameN  # exact frame index
            B6.tStart = t  # local t and not account for scr refresh
            B6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B6.started')
            # update status
            B6.status = STARTED
            win.callOnFlip(B6.buttonClock.reset)
            B6.setAutoDraw(True)
        
        # if B6 is active this frame...
        if B6.status == STARTED:
            # update params
            pass
            # check whether B6 has been pressed
            if B6.isClicked:
                if not B6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B6.timesOn.append(B6.buttonClock.getTime())
                    B6.timesOff.append(B6.buttonClock.getTime())
                elif len(B6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B6.timesOff[-1] = B6.buttonClock.getTime()
                if not B6.wasClicked:
                    # end routine when B6 is clicked
                    continueRoutine = False
                if not B6.wasClicked:
                    # run callback code when B6 is clicked
                    pass
        # take note of whether B6 was clicked, so that next frame we know if clicks are new
        B6.wasClicked = B6.isClicked and B6.status == STARTED
        # *B7* updates
        
        # if B7 is starting this frame...
        if B7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B7.frameNStart = frameN  # exact frame index
            B7.tStart = t  # local t and not account for scr refresh
            B7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B7.started')
            # update status
            B7.status = STARTED
            win.callOnFlip(B7.buttonClock.reset)
            B7.setAutoDraw(True)
        
        # if B7 is active this frame...
        if B7.status == STARTED:
            # update params
            pass
            # check whether B7 has been pressed
            if B7.isClicked:
                if not B7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B7.timesOn.append(B7.buttonClock.getTime())
                    B7.timesOff.append(B7.buttonClock.getTime())
                elif len(B7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B7.timesOff[-1] = B7.buttonClock.getTime()
                if not B7.wasClicked:
                    # end routine when B7 is clicked
                    continueRoutine = False
                if not B7.wasClicked:
                    # run callback code when B7 is clicked
                    pass
        # take note of whether B7 was clicked, so that next frame we know if clicks are new
        B7.wasClicked = B7.isClicked and B7.status == STARTED
        # *B8* updates
        
        # if B8 is starting this frame...
        if B8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B8.frameNStart = frameN  # exact frame index
            B8.tStart = t  # local t and not account for scr refresh
            B8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B8.started')
            # update status
            B8.status = STARTED
            win.callOnFlip(B8.buttonClock.reset)
            B8.setAutoDraw(True)
        
        # if B8 is active this frame...
        if B8.status == STARTED:
            # update params
            pass
            # check whether B8 has been pressed
            if B8.isClicked:
                if not B8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B8.timesOn.append(B8.buttonClock.getTime())
                    B8.timesOff.append(B8.buttonClock.getTime())
                elif len(B8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B8.timesOff[-1] = B8.buttonClock.getTime()
                if not B8.wasClicked:
                    # end routine when B8 is clicked
                    continueRoutine = False
                if not B8.wasClicked:
                    # run callback code when B8 is clicked
                    pass
        # take note of whether B8 was clicked, so that next frame we know if clicks are new
        B8.wasClicked = B8.isClicked and B8.status == STARTED
        # *V1* updates
        
        # if V1 is starting this frame...
        if V1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V1.frameNStart = frameN  # exact frame index
            V1.tStart = t  # local t and not account for scr refresh
            V1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V1.started')
            # update status
            V1.status = STARTED
            win.callOnFlip(V1.buttonClock.reset)
            V1.setAutoDraw(True)
        
        # if V1 is active this frame...
        if V1.status == STARTED:
            # update params
            pass
            # check whether V1 has been pressed
            if V1.isClicked:
                if not V1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V1.timesOn.append(V1.buttonClock.getTime())
                    V1.timesOff.append(V1.buttonClock.getTime())
                elif len(V1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V1.timesOff[-1] = V1.buttonClock.getTime()
                if not V1.wasClicked:
                    # end routine when V1 is clicked
                    continueRoutine = False
                if not V1.wasClicked:
                    # run callback code when V1 is clicked
                    pass
        # take note of whether V1 was clicked, so that next frame we know if clicks are new
        V1.wasClicked = V1.isClicked and V1.status == STARTED
        # *V2* updates
        
        # if V2 is starting this frame...
        if V2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V2.frameNStart = frameN  # exact frame index
            V2.tStart = t  # local t and not account for scr refresh
            V2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V2.started')
            # update status
            V2.status = STARTED
            win.callOnFlip(V2.buttonClock.reset)
            V2.setAutoDraw(True)
        
        # if V2 is active this frame...
        if V2.status == STARTED:
            # update params
            pass
            # check whether V2 has been pressed
            if V2.isClicked:
                if not V2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V2.timesOn.append(V2.buttonClock.getTime())
                    V2.timesOff.append(V2.buttonClock.getTime())
                elif len(V2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V2.timesOff[-1] = V2.buttonClock.getTime()
                if not V2.wasClicked:
                    # end routine when V2 is clicked
                    continueRoutine = False
                if not V2.wasClicked:
                    # run callback code when V2 is clicked
                    pass
        # take note of whether V2 was clicked, so that next frame we know if clicks are new
        V2.wasClicked = V2.isClicked and V2.status == STARTED
        # *V3* updates
        
        # if V3 is starting this frame...
        if V3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V3.frameNStart = frameN  # exact frame index
            V3.tStart = t  # local t and not account for scr refresh
            V3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V3.started')
            # update status
            V3.status = STARTED
            win.callOnFlip(V3.buttonClock.reset)
            V3.setAutoDraw(True)
        
        # if V3 is active this frame...
        if V3.status == STARTED:
            # update params
            pass
            # check whether V3 has been pressed
            if V3.isClicked:
                if not V3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V3.timesOn.append(V3.buttonClock.getTime())
                    V3.timesOff.append(V3.buttonClock.getTime())
                elif len(V3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V3.timesOff[-1] = V3.buttonClock.getTime()
                if not V3.wasClicked:
                    # end routine when V3 is clicked
                    continueRoutine = False
                if not V3.wasClicked:
                    # run callback code when V3 is clicked
                    pass
        # take note of whether V3 was clicked, so that next frame we know if clicks are new
        V3.wasClicked = V3.isClicked and V3.status == STARTED
        # *V4* updates
        
        # if V4 is starting this frame...
        if V4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V4.frameNStart = frameN  # exact frame index
            V4.tStart = t  # local t and not account for scr refresh
            V4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V4.started')
            # update status
            V4.status = STARTED
            win.callOnFlip(V4.buttonClock.reset)
            V4.setAutoDraw(True)
        
        # if V4 is active this frame...
        if V4.status == STARTED:
            # update params
            pass
            # check whether V4 has been pressed
            if V4.isClicked:
                if not V4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V4.timesOn.append(V4.buttonClock.getTime())
                    V4.timesOff.append(V4.buttonClock.getTime())
                elif len(V4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V4.timesOff[-1] = V4.buttonClock.getTime()
                if not V4.wasClicked:
                    # end routine when V4 is clicked
                    continueRoutine = False
                if not V4.wasClicked:
                    # run callback code when V4 is clicked
                    pass
        # take note of whether V4 was clicked, so that next frame we know if clicks are new
        V4.wasClicked = V4.isClicked and V4.status == STARTED
        # *V5* updates
        
        # if V5 is starting this frame...
        if V5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V5.frameNStart = frameN  # exact frame index
            V5.tStart = t  # local t and not account for scr refresh
            V5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V5.started')
            # update status
            V5.status = STARTED
            win.callOnFlip(V5.buttonClock.reset)
            V5.setAutoDraw(True)
        
        # if V5 is active this frame...
        if V5.status == STARTED:
            # update params
            pass
            # check whether V5 has been pressed
            if V5.isClicked:
                if not V5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V5.timesOn.append(V5.buttonClock.getTime())
                    V5.timesOff.append(V5.buttonClock.getTime())
                elif len(V5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V5.timesOff[-1] = V5.buttonClock.getTime()
                if not V5.wasClicked:
                    # end routine when V5 is clicked
                    continueRoutine = False
                if not V5.wasClicked:
                    # run callback code when V5 is clicked
                    pass
        # take note of whether V5 was clicked, so that next frame we know if clicks are new
        V5.wasClicked = V5.isClicked and V5.status == STARTED
        # *V6* updates
        
        # if V6 is starting this frame...
        if V6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V6.frameNStart = frameN  # exact frame index
            V6.tStart = t  # local t and not account for scr refresh
            V6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V6.started')
            # update status
            V6.status = STARTED
            win.callOnFlip(V6.buttonClock.reset)
            V6.setAutoDraw(True)
        
        # if V6 is active this frame...
        if V6.status == STARTED:
            # update params
            pass
            # check whether V6 has been pressed
            if V6.isClicked:
                if not V6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V6.timesOn.append(V6.buttonClock.getTime())
                    V6.timesOff.append(V6.buttonClock.getTime())
                elif len(V6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V6.timesOff[-1] = V6.buttonClock.getTime()
                if not V6.wasClicked:
                    # end routine when V6 is clicked
                    continueRoutine = False
                if not V6.wasClicked:
                    # run callback code when V6 is clicked
                    pass
        # take note of whether V6 was clicked, so that next frame we know if clicks are new
        V6.wasClicked = V6.isClicked and V6.status == STARTED
        # *V7* updates
        
        # if V7 is starting this frame...
        if V7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V7.frameNStart = frameN  # exact frame index
            V7.tStart = t  # local t and not account for scr refresh
            V7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V7.started')
            # update status
            V7.status = STARTED
            win.callOnFlip(V7.buttonClock.reset)
            V7.setAutoDraw(True)
        
        # if V7 is active this frame...
        if V7.status == STARTED:
            # update params
            pass
            # check whether V7 has been pressed
            if V7.isClicked:
                if not V7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V7.timesOn.append(V7.buttonClock.getTime())
                    V7.timesOff.append(V7.buttonClock.getTime())
                elif len(V7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V7.timesOff[-1] = V7.buttonClock.getTime()
                if not V7.wasClicked:
                    # end routine when V7 is clicked
                    continueRoutine = False
                if not V7.wasClicked:
                    # run callback code when V7 is clicked
                    pass
        # take note of whether V7 was clicked, so that next frame we know if clicks are new
        V7.wasClicked = V7.isClicked and V7.status == STARTED
        # *V8* updates
        
        # if V8 is starting this frame...
        if V8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V8.frameNStart = frameN  # exact frame index
            V8.tStart = t  # local t and not account for scr refresh
            V8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V8.started')
            # update status
            V8.status = STARTED
            win.callOnFlip(V8.buttonClock.reset)
            V8.setAutoDraw(True)
        
        # if V8 is active this frame...
        if V8.status == STARTED:
            # update params
            pass
            # check whether V8 has been pressed
            if V8.isClicked:
                if not V8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V8.timesOn.append(V8.buttonClock.getTime())
                    V8.timesOff.append(V8.buttonClock.getTime())
                elif len(V8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V8.timesOff[-1] = V8.buttonClock.getTime()
                if not V8.wasClicked:
                    # end routine when V8 is clicked
                    continueRoutine = False
                if not V8.wasClicked:
                    # run callback code when V8 is clicked
                    pass
        # take note of whether V8 was clicked, so that next frame we know if clicks are new
        V8.wasClicked = V8.isClicked and V8.status == STARTED
        # *J1* updates
        
        # if J1 is starting this frame...
        if J1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J1.frameNStart = frameN  # exact frame index
            J1.tStart = t  # local t and not account for scr refresh
            J1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J1.started')
            # update status
            J1.status = STARTED
            win.callOnFlip(J1.buttonClock.reset)
            J1.setAutoDraw(True)
        
        # if J1 is active this frame...
        if J1.status == STARTED:
            # update params
            pass
            # check whether J1 has been pressed
            if J1.isClicked:
                if not J1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J1.timesOn.append(J1.buttonClock.getTime())
                    J1.timesOff.append(J1.buttonClock.getTime())
                elif len(J1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J1.timesOff[-1] = J1.buttonClock.getTime()
                if not J1.wasClicked:
                    # end routine when J1 is clicked
                    continueRoutine = False
                if not J1.wasClicked:
                    # run callback code when J1 is clicked
                    pass
        # take note of whether J1 was clicked, so that next frame we know if clicks are new
        J1.wasClicked = J1.isClicked and J1.status == STARTED
        # *J2* updates
        
        # if J2 is starting this frame...
        if J2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J2.frameNStart = frameN  # exact frame index
            J2.tStart = t  # local t and not account for scr refresh
            J2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J2.started')
            # update status
            J2.status = STARTED
            win.callOnFlip(J2.buttonClock.reset)
            J2.setAutoDraw(True)
        
        # if J2 is active this frame...
        if J2.status == STARTED:
            # update params
            pass
            # check whether J2 has been pressed
            if J2.isClicked:
                if not J2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J2.timesOn.append(J2.buttonClock.getTime())
                    J2.timesOff.append(J2.buttonClock.getTime())
                elif len(J2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J2.timesOff[-1] = J2.buttonClock.getTime()
                if not J2.wasClicked:
                    # end routine when J2 is clicked
                    continueRoutine = False
                if not J2.wasClicked:
                    # run callback code when J2 is clicked
                    pass
        # take note of whether J2 was clicked, so that next frame we know if clicks are new
        J2.wasClicked = J2.isClicked and J2.status == STARTED
        # *J3* updates
        
        # if J3 is starting this frame...
        if J3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J3.frameNStart = frameN  # exact frame index
            J3.tStart = t  # local t and not account for scr refresh
            J3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J3.started')
            # update status
            J3.status = STARTED
            win.callOnFlip(J3.buttonClock.reset)
            J3.setAutoDraw(True)
        
        # if J3 is active this frame...
        if J3.status == STARTED:
            # update params
            pass
            # check whether J3 has been pressed
            if J3.isClicked:
                if not J3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J3.timesOn.append(J3.buttonClock.getTime())
                    J3.timesOff.append(J3.buttonClock.getTime())
                elif len(J3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J3.timesOff[-1] = J3.buttonClock.getTime()
                if not J3.wasClicked:
                    # end routine when J3 is clicked
                    continueRoutine = False
                if not J3.wasClicked:
                    # run callback code when J3 is clicked
                    pass
        # take note of whether J3 was clicked, so that next frame we know if clicks are new
        J3.wasClicked = J3.isClicked and J3.status == STARTED
        # *J4* updates
        
        # if J4 is starting this frame...
        if J4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J4.frameNStart = frameN  # exact frame index
            J4.tStart = t  # local t and not account for scr refresh
            J4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J4.started')
            # update status
            J4.status = STARTED
            win.callOnFlip(J4.buttonClock.reset)
            J4.setAutoDraw(True)
        
        # if J4 is active this frame...
        if J4.status == STARTED:
            # update params
            pass
            # check whether J4 has been pressed
            if J4.isClicked:
                if not J4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J4.timesOn.append(J4.buttonClock.getTime())
                    J4.timesOff.append(J4.buttonClock.getTime())
                elif len(J4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J4.timesOff[-1] = J4.buttonClock.getTime()
                if not J4.wasClicked:
                    # end routine when J4 is clicked
                    continueRoutine = False
                if not J4.wasClicked:
                    # run callback code when J4 is clicked
                    pass
        # take note of whether J4 was clicked, so that next frame we know if clicks are new
        J4.wasClicked = J4.isClicked and J4.status == STARTED
        # *J5* updates
        
        # if J5 is starting this frame...
        if J5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J5.frameNStart = frameN  # exact frame index
            J5.tStart = t  # local t and not account for scr refresh
            J5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J5.started')
            # update status
            J5.status = STARTED
            win.callOnFlip(J5.buttonClock.reset)
            J5.setAutoDraw(True)
        
        # if J5 is active this frame...
        if J5.status == STARTED:
            # update params
            pass
            # check whether J5 has been pressed
            if J5.isClicked:
                if not J5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J5.timesOn.append(J5.buttonClock.getTime())
                    J5.timesOff.append(J5.buttonClock.getTime())
                elif len(J5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J5.timesOff[-1] = J5.buttonClock.getTime()
                if not J5.wasClicked:
                    # end routine when J5 is clicked
                    continueRoutine = False
                if not J5.wasClicked:
                    # run callback code when J5 is clicked
                    pass
        # take note of whether J5 was clicked, so that next frame we know if clicks are new
        J5.wasClicked = J5.isClicked and J5.status == STARTED
        # *J6* updates
        
        # if J6 is starting this frame...
        if J6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J6.frameNStart = frameN  # exact frame index
            J6.tStart = t  # local t and not account for scr refresh
            J6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J6.started')
            # update status
            J6.status = STARTED
            win.callOnFlip(J6.buttonClock.reset)
            J6.setAutoDraw(True)
        
        # if J6 is active this frame...
        if J6.status == STARTED:
            # update params
            pass
            # check whether J6 has been pressed
            if J6.isClicked:
                if not J6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J6.timesOn.append(J6.buttonClock.getTime())
                    J6.timesOff.append(J6.buttonClock.getTime())
                elif len(J6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J6.timesOff[-1] = J6.buttonClock.getTime()
                if not J6.wasClicked:
                    # end routine when J6 is clicked
                    continueRoutine = False
                if not J6.wasClicked:
                    # run callback code when J6 is clicked
                    pass
        # take note of whether J6 was clicked, so that next frame we know if clicks are new
        J6.wasClicked = J6.isClicked and J6.status == STARTED
        # *J7* updates
        
        # if J7 is starting this frame...
        if J7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J7.frameNStart = frameN  # exact frame index
            J7.tStart = t  # local t and not account for scr refresh
            J7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J7.started')
            # update status
            J7.status = STARTED
            win.callOnFlip(J7.buttonClock.reset)
            J7.setAutoDraw(True)
        
        # if J7 is active this frame...
        if J7.status == STARTED:
            # update params
            pass
            # check whether J7 has been pressed
            if J7.isClicked:
                if not J7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J7.timesOn.append(J7.buttonClock.getTime())
                    J7.timesOff.append(J7.buttonClock.getTime())
                elif len(J7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J7.timesOff[-1] = J7.buttonClock.getTime()
                if not J7.wasClicked:
                    # end routine when J7 is clicked
                    continueRoutine = False
                if not J7.wasClicked:
                    # run callback code when J7 is clicked
                    pass
        # take note of whether J7 was clicked, so that next frame we know if clicks are new
        J7.wasClicked = J7.isClicked and J7.status == STARTED
        # *J8* updates
        
        # if J8 is starting this frame...
        if J8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J8.frameNStart = frameN  # exact frame index
            J8.tStart = t  # local t and not account for scr refresh
            J8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J8.started')
            # update status
            J8.status = STARTED
            win.callOnFlip(J8.buttonClock.reset)
            J8.setAutoDraw(True)
        
        # if J8 is active this frame...
        if J8.status == STARTED:
            # update params
            pass
            # check whether J8 has been pressed
            if J8.isClicked:
                if not J8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J8.timesOn.append(J8.buttonClock.getTime())
                    J8.timesOff.append(J8.buttonClock.getTime())
                elif len(J8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J8.timesOff[-1] = J8.buttonClock.getTime()
                if not J8.wasClicked:
                    # end routine when J8 is clicked
                    continueRoutine = False
                if not J8.wasClicked:
                    # run callback code when J8 is clicked
                    pass
        # take note of whether J8 was clicked, so that next frame we know if clicks are new
        J8.wasClicked = J8.isClicked and J8.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=trial,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            trial.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if trial.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in trial.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial" ---
    for thisComponent in trial.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for trial
    trial.tStop = globalClock.getTime(format='float')
    trial.tStopRefresh = tThisFlipGlobal
    thisExp.addData('trial.stopped', trial.tStop)
    thisExp.addData('R1.numClicks', R1.numClicks)
    if R1.numClicks:
       thisExp.addData('R1.timesOn', R1.timesOn)
       thisExp.addData('R1.timesOff', R1.timesOff)
    else:
       thisExp.addData('R1.timesOn', "")
       thisExp.addData('R1.timesOff', "")
    thisExp.addData('R2.numClicks', R2.numClicks)
    if R2.numClicks:
       thisExp.addData('R2.timesOn', R2.timesOn)
       thisExp.addData('R2.timesOff', R2.timesOff)
    else:
       thisExp.addData('R2.timesOn', "")
       thisExp.addData('R2.timesOff', "")
    thisExp.addData('R3.numClicks', R3.numClicks)
    if R3.numClicks:
       thisExp.addData('R3.timesOn', R3.timesOn)
       thisExp.addData('R3.timesOff', R3.timesOff)
    else:
       thisExp.addData('R3.timesOn', "")
       thisExp.addData('R3.timesOff', "")
    thisExp.addData('R4.numClicks', R4.numClicks)
    if R4.numClicks:
       thisExp.addData('R4.timesOn', R4.timesOn)
       thisExp.addData('R4.timesOff', R4.timesOff)
    else:
       thisExp.addData('R4.timesOn', "")
       thisExp.addData('R4.timesOff', "")
    thisExp.addData('R5.numClicks', R5.numClicks)
    if R5.numClicks:
       thisExp.addData('R5.timesOn', R5.timesOn)
       thisExp.addData('R5.timesOff', R5.timesOff)
    else:
       thisExp.addData('R5.timesOn', "")
       thisExp.addData('R5.timesOff', "")
    thisExp.addData('R6.numClicks', R6.numClicks)
    if R6.numClicks:
       thisExp.addData('R6.timesOn', R6.timesOn)
       thisExp.addData('R6.timesOff', R6.timesOff)
    else:
       thisExp.addData('R6.timesOn', "")
       thisExp.addData('R6.timesOff', "")
    thisExp.addData('R7.numClicks', R7.numClicks)
    if R7.numClicks:
       thisExp.addData('R7.timesOn', R7.timesOn)
       thisExp.addData('R7.timesOff', R7.timesOff)
    else:
       thisExp.addData('R7.timesOn', "")
       thisExp.addData('R7.timesOff', "")
    thisExp.addData('R8.numClicks', R8.numClicks)
    if R8.numClicks:
       thisExp.addData('R8.timesOn', R8.timesOn)
       thisExp.addData('R8.timesOff', R8.timesOff)
    else:
       thisExp.addData('R8.timesOn', "")
       thisExp.addData('R8.timesOff', "")
    thisExp.addData('B1.numClicks', B1.numClicks)
    if B1.numClicks:
       thisExp.addData('B1.timesOn', B1.timesOn)
       thisExp.addData('B1.timesOff', B1.timesOff)
    else:
       thisExp.addData('B1.timesOn', "")
       thisExp.addData('B1.timesOff', "")
    thisExp.addData('B2.numClicks', B2.numClicks)
    if B2.numClicks:
       thisExp.addData('B2.timesOn', B2.timesOn)
       thisExp.addData('B2.timesOff', B2.timesOff)
    else:
       thisExp.addData('B2.timesOn', "")
       thisExp.addData('B2.timesOff', "")
    thisExp.addData('B3.numClicks', B3.numClicks)
    if B3.numClicks:
       thisExp.addData('B3.timesOn', B3.timesOn)
       thisExp.addData('B3.timesOff', B3.timesOff)
    else:
       thisExp.addData('B3.timesOn', "")
       thisExp.addData('B3.timesOff', "")
    thisExp.addData('B4.numClicks', B4.numClicks)
    if B4.numClicks:
       thisExp.addData('B4.timesOn', B4.timesOn)
       thisExp.addData('B4.timesOff', B4.timesOff)
    else:
       thisExp.addData('B4.timesOn', "")
       thisExp.addData('B4.timesOff', "")
    thisExp.addData('B5.numClicks', B5.numClicks)
    if B5.numClicks:
       thisExp.addData('B5.timesOn', B5.timesOn)
       thisExp.addData('B5.timesOff', B5.timesOff)
    else:
       thisExp.addData('B5.timesOn', "")
       thisExp.addData('B5.timesOff', "")
    thisExp.addData('B6.numClicks', B6.numClicks)
    if B6.numClicks:
       thisExp.addData('B6.timesOn', B6.timesOn)
       thisExp.addData('B6.timesOff', B6.timesOff)
    else:
       thisExp.addData('B6.timesOn', "")
       thisExp.addData('B6.timesOff', "")
    thisExp.addData('B7.numClicks', B7.numClicks)
    if B7.numClicks:
       thisExp.addData('B7.timesOn', B7.timesOn)
       thisExp.addData('B7.timesOff', B7.timesOff)
    else:
       thisExp.addData('B7.timesOn', "")
       thisExp.addData('B7.timesOff', "")
    thisExp.addData('B8.numClicks', B8.numClicks)
    if B8.numClicks:
       thisExp.addData('B8.timesOn', B8.timesOn)
       thisExp.addData('B8.timesOff', B8.timesOff)
    else:
       thisExp.addData('B8.timesOn', "")
       thisExp.addData('B8.timesOff', "")
    thisExp.addData('V1.numClicks', V1.numClicks)
    if V1.numClicks:
       thisExp.addData('V1.timesOn', V1.timesOn)
       thisExp.addData('V1.timesOff', V1.timesOff)
    else:
       thisExp.addData('V1.timesOn', "")
       thisExp.addData('V1.timesOff', "")
    thisExp.addData('V2.numClicks', V2.numClicks)
    if V2.numClicks:
       thisExp.addData('V2.timesOn', V2.timesOn)
       thisExp.addData('V2.timesOff', V2.timesOff)
    else:
       thisExp.addData('V2.timesOn', "")
       thisExp.addData('V2.timesOff', "")
    thisExp.addData('V3.numClicks', V3.numClicks)
    if V3.numClicks:
       thisExp.addData('V3.timesOn', V3.timesOn)
       thisExp.addData('V3.timesOff', V3.timesOff)
    else:
       thisExp.addData('V3.timesOn', "")
       thisExp.addData('V3.timesOff', "")
    thisExp.addData('V4.numClicks', V4.numClicks)
    if V4.numClicks:
       thisExp.addData('V4.timesOn', V4.timesOn)
       thisExp.addData('V4.timesOff', V4.timesOff)
    else:
       thisExp.addData('V4.timesOn', "")
       thisExp.addData('V4.timesOff', "")
    thisExp.addData('V5.numClicks', V5.numClicks)
    if V5.numClicks:
       thisExp.addData('V5.timesOn', V5.timesOn)
       thisExp.addData('V5.timesOff', V5.timesOff)
    else:
       thisExp.addData('V5.timesOn', "")
       thisExp.addData('V5.timesOff', "")
    thisExp.addData('V6.numClicks', V6.numClicks)
    if V6.numClicks:
       thisExp.addData('V6.timesOn', V6.timesOn)
       thisExp.addData('V6.timesOff', V6.timesOff)
    else:
       thisExp.addData('V6.timesOn', "")
       thisExp.addData('V6.timesOff', "")
    thisExp.addData('V7.numClicks', V7.numClicks)
    if V7.numClicks:
       thisExp.addData('V7.timesOn', V7.timesOn)
       thisExp.addData('V7.timesOff', V7.timesOff)
    else:
       thisExp.addData('V7.timesOn', "")
       thisExp.addData('V7.timesOff', "")
    thisExp.addData('V8.numClicks', V8.numClicks)
    if V8.numClicks:
       thisExp.addData('V8.timesOn', V8.timesOn)
       thisExp.addData('V8.timesOff', V8.timesOff)
    else:
       thisExp.addData('V8.timesOn', "")
       thisExp.addData('V8.timesOff', "")
    thisExp.addData('J1.numClicks', J1.numClicks)
    if J1.numClicks:
       thisExp.addData('J1.timesOn', J1.timesOn)
       thisExp.addData('J1.timesOff', J1.timesOff)
    else:
       thisExp.addData('J1.timesOn', "")
       thisExp.addData('J1.timesOff', "")
    thisExp.addData('J2.numClicks', J2.numClicks)
    if J2.numClicks:
       thisExp.addData('J2.timesOn', J2.timesOn)
       thisExp.addData('J2.timesOff', J2.timesOff)
    else:
       thisExp.addData('J2.timesOn', "")
       thisExp.addData('J2.timesOff', "")
    thisExp.addData('J3.numClicks', J3.numClicks)
    if J3.numClicks:
       thisExp.addData('J3.timesOn', J3.timesOn)
       thisExp.addData('J3.timesOff', J3.timesOff)
    else:
       thisExp.addData('J3.timesOn', "")
       thisExp.addData('J3.timesOff', "")
    thisExp.addData('J4.numClicks', J4.numClicks)
    if J4.numClicks:
       thisExp.addData('J4.timesOn', J4.timesOn)
       thisExp.addData('J4.timesOff', J4.timesOff)
    else:
       thisExp.addData('J4.timesOn', "")
       thisExp.addData('J4.timesOff', "")
    thisExp.addData('J5.numClicks', J5.numClicks)
    if J5.numClicks:
       thisExp.addData('J5.timesOn', J5.timesOn)
       thisExp.addData('J5.timesOff', J5.timesOff)
    else:
       thisExp.addData('J5.timesOn', "")
       thisExp.addData('J5.timesOff', "")
    thisExp.addData('J6.numClicks', J6.numClicks)
    if J6.numClicks:
       thisExp.addData('J6.timesOn', J6.timesOn)
       thisExp.addData('J6.timesOff', J6.timesOff)
    else:
       thisExp.addData('J6.timesOn', "")
       thisExp.addData('J6.timesOff', "")
    thisExp.addData('J7.numClicks', J7.numClicks)
    if J7.numClicks:
       thisExp.addData('J7.timesOn', J7.timesOn)
       thisExp.addData('J7.timesOff', J7.timesOff)
    else:
       thisExp.addData('J7.timesOn', "")
       thisExp.addData('J7.timesOff', "")
    thisExp.addData('J8.numClicks', J8.numClicks)
    if J8.numClicks:
       thisExp.addData('J8.timesOn', J8.timesOn)
       thisExp.addData('J8.timesOff', J8.timesOff)
    else:
       thisExp.addData('J8.timesOn', "")
       thisExp.addData('J8.timesOff', "")
    thisExp.nextEntry()
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "escu" ---
    # create an object to store info about Routine escu
    escu = data.Routine(
        name='escu',
        components=[Pasdeffort, __1, Trespeudeffort],
    )
    escu.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset Pasdeffort to account for continued clicks & clear times on/off
    Pasdeffort.reset()
    # reset __1 to account for continued clicks & clear times on/off
    __1.reset()
    # reset Trespeudeffort to account for continued clicks & clear times on/off
    Trespeudeffort.reset()
    # store start times for escu
    escu.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    escu.tStart = globalClock.getTime(format='float')
    escu.status = STARTED
    thisExp.addData('escu.started', escu.tStart)
    escu.maxDuration = None
    # keep track of which components have finished
    escuComponents = escu.components
    for thisComponent in escu.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "escu" ---
    thisExp.currentRoutine = escu
    escu.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *Pasdeffort* updates
        
        # if Pasdeffort is starting this frame...
        if Pasdeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Pasdeffort.frameNStart = frameN  # exact frame index
            Pasdeffort.tStart = t  # local t and not account for scr refresh
            Pasdeffort.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Pasdeffort, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Pasdeffort.started')
            # update status
            Pasdeffort.status = STARTED
            win.callOnFlip(Pasdeffort.buttonClock.reset)
            Pasdeffort.setAutoDraw(True)
        
        # if Pasdeffort is active this frame...
        if Pasdeffort.status == STARTED:
            # update params
            pass
            # check whether Pasdeffort has been pressed
            if Pasdeffort.isClicked:
                if not Pasdeffort.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    Pasdeffort.timesOn.append(Pasdeffort.buttonClock.getTime())
                    Pasdeffort.timesOff.append(Pasdeffort.buttonClock.getTime())
                elif len(Pasdeffort.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    Pasdeffort.timesOff[-1] = Pasdeffort.buttonClock.getTime()
                if not Pasdeffort.wasClicked:
                    # end routine when Pasdeffort is clicked
                    continueRoutine = False
                if not Pasdeffort.wasClicked:
                    # run callback code when Pasdeffort is clicked
                    pass
        # take note of whether Pasdeffort was clicked, so that next frame we know if clicks are new
        Pasdeffort.wasClicked = Pasdeffort.isClicked and Pasdeffort.status == STARTED
        # *__1* updates
        
        # if __1 is starting this frame...
        if __1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            __1.frameNStart = frameN  # exact frame index
            __1.tStart = t  # local t and not account for scr refresh
            __1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(__1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, '__1.started')
            # update status
            __1.status = STARTED
            win.callOnFlip(__1.buttonClock.reset)
            __1.setAutoDraw(True)
        
        # if __1 is active this frame...
        if __1.status == STARTED:
            # update params
            pass
            # check whether __1 has been pressed
            if __1.isClicked:
                if not __1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    __1.timesOn.append(__1.buttonClock.getTime())
                    __1.timesOff.append(__1.buttonClock.getTime())
                elif len(__1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    __1.timesOff[-1] = __1.buttonClock.getTime()
                if not __1.wasClicked:
                    # end routine when __1 is clicked
                    continueRoutine = False
                if not __1.wasClicked:
                    # run callback code when __1 is clicked
                    pass
        # take note of whether __1 was clicked, so that next frame we know if clicks are new
        __1.wasClicked = __1.isClicked and __1.status == STARTED
        # *Trespeudeffort* updates
        
        # if Trespeudeffort is starting this frame...
        if Trespeudeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Trespeudeffort.frameNStart = frameN  # exact frame index
            Trespeudeffort.tStart = t  # local t and not account for scr refresh
            Trespeudeffort.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Trespeudeffort, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Trespeudeffort.started')
            # update status
            Trespeudeffort.status = STARTED
            win.callOnFlip(Trespeudeffort.buttonClock.reset)
            Trespeudeffort.setAutoDraw(True)
        
        # if Trespeudeffort is active this frame...
        if Trespeudeffort.status == STARTED:
            # update params
            pass
            # check whether Trespeudeffort has been pressed
            if Trespeudeffort.isClicked:
                if not Trespeudeffort.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    Trespeudeffort.timesOn.append(Trespeudeffort.buttonClock.getTime())
                    Trespeudeffort.timesOff.append(Trespeudeffort.buttonClock.getTime())
                elif len(Trespeudeffort.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    Trespeudeffort.timesOff[-1] = Trespeudeffort.buttonClock.getTime()
                if not Trespeudeffort.wasClicked:
                    # end routine when Trespeudeffort is clicked
                    continueRoutine = False
                if not Trespeudeffort.wasClicked:
                    # run callback code when Trespeudeffort is clicked
                    pass
        # take note of whether Trespeudeffort was clicked, so that next frame we know if clicks are new
        Trespeudeffort.wasClicked = Trespeudeffort.isClicked and Trespeudeffort.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=escu,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            escu.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if escu.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in escu.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "escu" ---
    for thisComponent in escu.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for escu
    escu.tStop = globalClock.getTime(format='float')
    escu.tStopRefresh = tThisFlipGlobal
    thisExp.addData('escu.stopped', escu.tStop)
    thisExp.addData('Pasdeffort.numClicks', Pasdeffort.numClicks)
    if Pasdeffort.numClicks:
       thisExp.addData('Pasdeffort.timesOn', Pasdeffort.timesOn)
       thisExp.addData('Pasdeffort.timesOff', Pasdeffort.timesOff)
    else:
       thisExp.addData('Pasdeffort.timesOn', "")
       thisExp.addData('Pasdeffort.timesOff', "")
    thisExp.addData('__1.numClicks', __1.numClicks)
    if __1.numClicks:
       thisExp.addData('__1.timesOn', __1.timesOn)
       thisExp.addData('__1.timesOff', __1.timesOff)
    else:
       thisExp.addData('__1.timesOn', "")
       thisExp.addData('__1.timesOff', "")
    thisExp.addData('Trespeudeffort.numClicks', Trespeudeffort.numClicks)
    if Trespeudeffort.numClicks:
       thisExp.addData('Trespeudeffort.timesOn', Trespeudeffort.timesOn)
       thisExp.addData('Trespeudeffort.timesOff', Trespeudeffort.timesOff)
    else:
       thisExp.addData('Trespeudeffort.timesOn', "")
       thisExp.addData('Trespeudeffort.timesOff', "")
    thisExp.nextEntry()
    # the Routine "escu" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "trial" ---
    # create an object to store info about Routine trial
    trial = data.Routine(
        name='trial',
        components=[text_instruction, R1, R2, R3, R4, R5, R6, R7, R8, B1, B2, B3, B4, B5, B6, B7, B8, V1, V2, V3, V4, V5, V6, V7, V8, J1, J2, J3, J4, J5, J6, J7, J8],
    )
    trial.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset R1 to account for continued clicks & clear times on/off
    R1.reset()
    # reset R2 to account for continued clicks & clear times on/off
    R2.reset()
    # reset R3 to account for continued clicks & clear times on/off
    R3.reset()
    # reset R4 to account for continued clicks & clear times on/off
    R4.reset()
    # reset R5 to account for continued clicks & clear times on/off
    R5.reset()
    # reset R6 to account for continued clicks & clear times on/off
    R6.reset()
    # reset R7 to account for continued clicks & clear times on/off
    R7.reset()
    # reset R8 to account for continued clicks & clear times on/off
    R8.reset()
    # reset B1 to account for continued clicks & clear times on/off
    B1.reset()
    # reset B2 to account for continued clicks & clear times on/off
    B2.reset()
    # reset B3 to account for continued clicks & clear times on/off
    B3.reset()
    # reset B4 to account for continued clicks & clear times on/off
    B4.reset()
    # reset B5 to account for continued clicks & clear times on/off
    B5.reset()
    # reset B6 to account for continued clicks & clear times on/off
    B6.reset()
    # reset B7 to account for continued clicks & clear times on/off
    B7.reset()
    # reset B8 to account for continued clicks & clear times on/off
    B8.reset()
    # reset V1 to account for continued clicks & clear times on/off
    V1.reset()
    # reset V2 to account for continued clicks & clear times on/off
    V2.reset()
    # reset V3 to account for continued clicks & clear times on/off
    V3.reset()
    # reset V4 to account for continued clicks & clear times on/off
    V4.reset()
    # reset V5 to account for continued clicks & clear times on/off
    V5.reset()
    # reset V6 to account for continued clicks & clear times on/off
    V6.reset()
    # reset V7 to account for continued clicks & clear times on/off
    V7.reset()
    # reset V8 to account for continued clicks & clear times on/off
    V8.reset()
    # reset J1 to account for continued clicks & clear times on/off
    J1.reset()
    # reset J2 to account for continued clicks & clear times on/off
    J2.reset()
    # reset J3 to account for continued clicks & clear times on/off
    J3.reset()
    # reset J4 to account for continued clicks & clear times on/off
    J4.reset()
    # reset J5 to account for continued clicks & clear times on/off
    J5.reset()
    # reset J6 to account for continued clicks & clear times on/off
    J6.reset()
    # reset J7 to account for continued clicks & clear times on/off
    J7.reset()
    # reset J8 to account for continued clicks & clear times on/off
    J8.reset()
    # store start times for trial
    trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    trial.tStart = globalClock.getTime(format='float')
    trial.status = STARTED
    thisExp.addData('trial.started', trial.tStart)
    trial.maxDuration = None
    # keep track of which components have finished
    trialComponents = trial.components
    for thisComponent in trial.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial" ---
    thisExp.currentRoutine = trial
    trial.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruction* updates
        
        # if text_instruction is starting this frame...
        if text_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction.frameNStart = frameN  # exact frame index
            text_instruction.tStart = t  # local t and not account for scr refresh
            text_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction.started')
            # update status
            text_instruction.status = STARTED
            text_instruction.setAutoDraw(True)
        
        # if text_instruction is active this frame...
        if text_instruction.status == STARTED:
            # update params
            pass
        
        # if text_instruction is stopping this frame...
        if text_instruction.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_instruction.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_instruction.tStop = t  # not accounting for scr refresh
                text_instruction.tStopRefresh = tThisFlipGlobal  # on global time
                text_instruction.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_instruction.stopped')
                # update status
                text_instruction.status = FINISHED
                text_instruction.setAutoDraw(False)
        # *R1* updates
        
        # if R1 is starting this frame...
        if R1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R1.frameNStart = frameN  # exact frame index
            R1.tStart = t  # local t and not account for scr refresh
            R1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R1.started')
            # update status
            R1.status = STARTED
            win.callOnFlip(R1.buttonClock.reset)
            R1.setAutoDraw(True)
        
        # if R1 is active this frame...
        if R1.status == STARTED:
            # update params
            pass
            # check whether R1 has been pressed
            if R1.isClicked:
                if not R1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R1.timesOn.append(R1.buttonClock.getTime())
                    R1.timesOff.append(R1.buttonClock.getTime())
                elif len(R1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R1.timesOff[-1] = R1.buttonClock.getTime()
                if not R1.wasClicked:
                    # end routine when R1 is clicked
                    continueRoutine = False
                if not R1.wasClicked:
                    # run callback code when R1 is clicked
                    pass
        # take note of whether R1 was clicked, so that next frame we know if clicks are new
        R1.wasClicked = R1.isClicked and R1.status == STARTED
        # *R2* updates
        
        # if R2 is starting this frame...
        if R2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R2.frameNStart = frameN  # exact frame index
            R2.tStart = t  # local t and not account for scr refresh
            R2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R2.started')
            # update status
            R2.status = STARTED
            win.callOnFlip(R2.buttonClock.reset)
            R2.setAutoDraw(True)
        
        # if R2 is active this frame...
        if R2.status == STARTED:
            # update params
            pass
            # check whether R2 has been pressed
            if R2.isClicked:
                if not R2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R2.timesOn.append(R2.buttonClock.getTime())
                    R2.timesOff.append(R2.buttonClock.getTime())
                elif len(R2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R2.timesOff[-1] = R2.buttonClock.getTime()
                if not R2.wasClicked:
                    # end routine when R2 is clicked
                    continueRoutine = False
                if not R2.wasClicked:
                    # run callback code when R2 is clicked
                    pass
        # take note of whether R2 was clicked, so that next frame we know if clicks are new
        R2.wasClicked = R2.isClicked and R2.status == STARTED
        # *R3* updates
        
        # if R3 is starting this frame...
        if R3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R3.frameNStart = frameN  # exact frame index
            R3.tStart = t  # local t and not account for scr refresh
            R3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R3.started')
            # update status
            R3.status = STARTED
            win.callOnFlip(R3.buttonClock.reset)
            R3.setAutoDraw(True)
        
        # if R3 is active this frame...
        if R3.status == STARTED:
            # update params
            pass
            # check whether R3 has been pressed
            if R3.isClicked:
                if not R3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R3.timesOn.append(R3.buttonClock.getTime())
                    R3.timesOff.append(R3.buttonClock.getTime())
                elif len(R3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R3.timesOff[-1] = R3.buttonClock.getTime()
                if not R3.wasClicked:
                    # end routine when R3 is clicked
                    continueRoutine = False
                if not R3.wasClicked:
                    # run callback code when R3 is clicked
                    pass
        # take note of whether R3 was clicked, so that next frame we know if clicks are new
        R3.wasClicked = R3.isClicked and R3.status == STARTED
        # *R4* updates
        
        # if R4 is starting this frame...
        if R4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R4.frameNStart = frameN  # exact frame index
            R4.tStart = t  # local t and not account for scr refresh
            R4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R4.started')
            # update status
            R4.status = STARTED
            win.callOnFlip(R4.buttonClock.reset)
            R4.setAutoDraw(True)
        
        # if R4 is active this frame...
        if R4.status == STARTED:
            # update params
            pass
            # check whether R4 has been pressed
            if R4.isClicked:
                if not R4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R4.timesOn.append(R4.buttonClock.getTime())
                    R4.timesOff.append(R4.buttonClock.getTime())
                elif len(R4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R4.timesOff[-1] = R4.buttonClock.getTime()
                if not R4.wasClicked:
                    # end routine when R4 is clicked
                    continueRoutine = False
                if not R4.wasClicked:
                    # run callback code when R4 is clicked
                    pass
        # take note of whether R4 was clicked, so that next frame we know if clicks are new
        R4.wasClicked = R4.isClicked and R4.status == STARTED
        # *R5* updates
        
        # if R5 is starting this frame...
        if R5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R5.frameNStart = frameN  # exact frame index
            R5.tStart = t  # local t and not account for scr refresh
            R5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R5.started')
            # update status
            R5.status = STARTED
            win.callOnFlip(R5.buttonClock.reset)
            R5.setAutoDraw(True)
        
        # if R5 is active this frame...
        if R5.status == STARTED:
            # update params
            pass
            # check whether R5 has been pressed
            if R5.isClicked:
                if not R5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R5.timesOn.append(R5.buttonClock.getTime())
                    R5.timesOff.append(R5.buttonClock.getTime())
                elif len(R5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R5.timesOff[-1] = R5.buttonClock.getTime()
                if not R5.wasClicked:
                    # end routine when R5 is clicked
                    continueRoutine = False
                if not R5.wasClicked:
                    # run callback code when R5 is clicked
                    pass
        # take note of whether R5 was clicked, so that next frame we know if clicks are new
        R5.wasClicked = R5.isClicked and R5.status == STARTED
        # *R6* updates
        
        # if R6 is starting this frame...
        if R6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R6.frameNStart = frameN  # exact frame index
            R6.tStart = t  # local t and not account for scr refresh
            R6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R6.started')
            # update status
            R6.status = STARTED
            win.callOnFlip(R6.buttonClock.reset)
            R6.setAutoDraw(True)
        
        # if R6 is active this frame...
        if R6.status == STARTED:
            # update params
            pass
            # check whether R6 has been pressed
            if R6.isClicked:
                if not R6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R6.timesOn.append(R6.buttonClock.getTime())
                    R6.timesOff.append(R6.buttonClock.getTime())
                elif len(R6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R6.timesOff[-1] = R6.buttonClock.getTime()
                if not R6.wasClicked:
                    # end routine when R6 is clicked
                    continueRoutine = False
                if not R6.wasClicked:
                    # run callback code when R6 is clicked
                    pass
        # take note of whether R6 was clicked, so that next frame we know if clicks are new
        R6.wasClicked = R6.isClicked and R6.status == STARTED
        # *R7* updates
        
        # if R7 is starting this frame...
        if R7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R7.frameNStart = frameN  # exact frame index
            R7.tStart = t  # local t and not account for scr refresh
            R7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R7.started')
            # update status
            R7.status = STARTED
            win.callOnFlip(R7.buttonClock.reset)
            R7.setAutoDraw(True)
        
        # if R7 is active this frame...
        if R7.status == STARTED:
            # update params
            pass
            # check whether R7 has been pressed
            if R7.isClicked:
                if not R7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R7.timesOn.append(R7.buttonClock.getTime())
                    R7.timesOff.append(R7.buttonClock.getTime())
                elif len(R7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R7.timesOff[-1] = R7.buttonClock.getTime()
                if not R7.wasClicked:
                    # end routine when R7 is clicked
                    continueRoutine = False
                if not R7.wasClicked:
                    # run callback code when R7 is clicked
                    pass
        # take note of whether R7 was clicked, so that next frame we know if clicks are new
        R7.wasClicked = R7.isClicked and R7.status == STARTED
        # *R8* updates
        
        # if R8 is starting this frame...
        if R8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            R8.frameNStart = frameN  # exact frame index
            R8.tStart = t  # local t and not account for scr refresh
            R8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(R8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'R8.started')
            # update status
            R8.status = STARTED
            win.callOnFlip(R8.buttonClock.reset)
            R8.setAutoDraw(True)
        
        # if R8 is active this frame...
        if R8.status == STARTED:
            # update params
            pass
            # check whether R8 has been pressed
            if R8.isClicked:
                if not R8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    R8.timesOn.append(R8.buttonClock.getTime())
                    R8.timesOff.append(R8.buttonClock.getTime())
                elif len(R8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    R8.timesOff[-1] = R8.buttonClock.getTime()
                if not R8.wasClicked:
                    # end routine when R8 is clicked
                    continueRoutine = False
                if not R8.wasClicked:
                    # run callback code when R8 is clicked
                    pass
        # take note of whether R8 was clicked, so that next frame we know if clicks are new
        R8.wasClicked = R8.isClicked and R8.status == STARTED
        # *B1* updates
        
        # if B1 is starting this frame...
        if B1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B1.frameNStart = frameN  # exact frame index
            B1.tStart = t  # local t and not account for scr refresh
            B1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B1.started')
            # update status
            B1.status = STARTED
            win.callOnFlip(B1.buttonClock.reset)
            B1.setAutoDraw(True)
        
        # if B1 is active this frame...
        if B1.status == STARTED:
            # update params
            pass
            # check whether B1 has been pressed
            if B1.isClicked:
                if not B1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B1.timesOn.append(B1.buttonClock.getTime())
                    B1.timesOff.append(B1.buttonClock.getTime())
                elif len(B1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B1.timesOff[-1] = B1.buttonClock.getTime()
                if not B1.wasClicked:
                    # end routine when B1 is clicked
                    continueRoutine = False
                if not B1.wasClicked:
                    # run callback code when B1 is clicked
                    pass
        # take note of whether B1 was clicked, so that next frame we know if clicks are new
        B1.wasClicked = B1.isClicked and B1.status == STARTED
        # *B2* updates
        
        # if B2 is starting this frame...
        if B2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B2.frameNStart = frameN  # exact frame index
            B2.tStart = t  # local t and not account for scr refresh
            B2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B2.started')
            # update status
            B2.status = STARTED
            win.callOnFlip(B2.buttonClock.reset)
            B2.setAutoDraw(True)
        
        # if B2 is active this frame...
        if B2.status == STARTED:
            # update params
            pass
            # check whether B2 has been pressed
            if B2.isClicked:
                if not B2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B2.timesOn.append(B2.buttonClock.getTime())
                    B2.timesOff.append(B2.buttonClock.getTime())
                elif len(B2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B2.timesOff[-1] = B2.buttonClock.getTime()
                if not B2.wasClicked:
                    # end routine when B2 is clicked
                    continueRoutine = False
                if not B2.wasClicked:
                    # run callback code when B2 is clicked
                    pass
        # take note of whether B2 was clicked, so that next frame we know if clicks are new
        B2.wasClicked = B2.isClicked and B2.status == STARTED
        # *B3* updates
        
        # if B3 is starting this frame...
        if B3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B3.frameNStart = frameN  # exact frame index
            B3.tStart = t  # local t and not account for scr refresh
            B3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B3.started')
            # update status
            B3.status = STARTED
            win.callOnFlip(B3.buttonClock.reset)
            B3.setAutoDraw(True)
        
        # if B3 is active this frame...
        if B3.status == STARTED:
            # update params
            pass
            # check whether B3 has been pressed
            if B3.isClicked:
                if not B3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B3.timesOn.append(B3.buttonClock.getTime())
                    B3.timesOff.append(B3.buttonClock.getTime())
                elif len(B3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B3.timesOff[-1] = B3.buttonClock.getTime()
                if not B3.wasClicked:
                    # end routine when B3 is clicked
                    continueRoutine = False
                if not B3.wasClicked:
                    # run callback code when B3 is clicked
                    pass
        # take note of whether B3 was clicked, so that next frame we know if clicks are new
        B3.wasClicked = B3.isClicked and B3.status == STARTED
        # *B4* updates
        
        # if B4 is starting this frame...
        if B4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B4.frameNStart = frameN  # exact frame index
            B4.tStart = t  # local t and not account for scr refresh
            B4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B4.started')
            # update status
            B4.status = STARTED
            win.callOnFlip(B4.buttonClock.reset)
            B4.setAutoDraw(True)
        
        # if B4 is active this frame...
        if B4.status == STARTED:
            # update params
            pass
            # check whether B4 has been pressed
            if B4.isClicked:
                if not B4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B4.timesOn.append(B4.buttonClock.getTime())
                    B4.timesOff.append(B4.buttonClock.getTime())
                elif len(B4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B4.timesOff[-1] = B4.buttonClock.getTime()
                if not B4.wasClicked:
                    # end routine when B4 is clicked
                    continueRoutine = False
                if not B4.wasClicked:
                    # run callback code when B4 is clicked
                    pass
        # take note of whether B4 was clicked, so that next frame we know if clicks are new
        B4.wasClicked = B4.isClicked and B4.status == STARTED
        # *B5* updates
        
        # if B5 is starting this frame...
        if B5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B5.frameNStart = frameN  # exact frame index
            B5.tStart = t  # local t and not account for scr refresh
            B5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B5.started')
            # update status
            B5.status = STARTED
            win.callOnFlip(B5.buttonClock.reset)
            B5.setAutoDraw(True)
        
        # if B5 is active this frame...
        if B5.status == STARTED:
            # update params
            pass
            # check whether B5 has been pressed
            if B5.isClicked:
                if not B5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B5.timesOn.append(B5.buttonClock.getTime())
                    B5.timesOff.append(B5.buttonClock.getTime())
                elif len(B5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B5.timesOff[-1] = B5.buttonClock.getTime()
                if not B5.wasClicked:
                    # end routine when B5 is clicked
                    continueRoutine = False
                if not B5.wasClicked:
                    # run callback code when B5 is clicked
                    pass
        # take note of whether B5 was clicked, so that next frame we know if clicks are new
        B5.wasClicked = B5.isClicked and B5.status == STARTED
        # *B6* updates
        
        # if B6 is starting this frame...
        if B6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B6.frameNStart = frameN  # exact frame index
            B6.tStart = t  # local t and not account for scr refresh
            B6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B6.started')
            # update status
            B6.status = STARTED
            win.callOnFlip(B6.buttonClock.reset)
            B6.setAutoDraw(True)
        
        # if B6 is active this frame...
        if B6.status == STARTED:
            # update params
            pass
            # check whether B6 has been pressed
            if B6.isClicked:
                if not B6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B6.timesOn.append(B6.buttonClock.getTime())
                    B6.timesOff.append(B6.buttonClock.getTime())
                elif len(B6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B6.timesOff[-1] = B6.buttonClock.getTime()
                if not B6.wasClicked:
                    # end routine when B6 is clicked
                    continueRoutine = False
                if not B6.wasClicked:
                    # run callback code when B6 is clicked
                    pass
        # take note of whether B6 was clicked, so that next frame we know if clicks are new
        B6.wasClicked = B6.isClicked and B6.status == STARTED
        # *B7* updates
        
        # if B7 is starting this frame...
        if B7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B7.frameNStart = frameN  # exact frame index
            B7.tStart = t  # local t and not account for scr refresh
            B7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B7.started')
            # update status
            B7.status = STARTED
            win.callOnFlip(B7.buttonClock.reset)
            B7.setAutoDraw(True)
        
        # if B7 is active this frame...
        if B7.status == STARTED:
            # update params
            pass
            # check whether B7 has been pressed
            if B7.isClicked:
                if not B7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B7.timesOn.append(B7.buttonClock.getTime())
                    B7.timesOff.append(B7.buttonClock.getTime())
                elif len(B7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B7.timesOff[-1] = B7.buttonClock.getTime()
                if not B7.wasClicked:
                    # end routine when B7 is clicked
                    continueRoutine = False
                if not B7.wasClicked:
                    # run callback code when B7 is clicked
                    pass
        # take note of whether B7 was clicked, so that next frame we know if clicks are new
        B7.wasClicked = B7.isClicked and B7.status == STARTED
        # *B8* updates
        
        # if B8 is starting this frame...
        if B8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            B8.frameNStart = frameN  # exact frame index
            B8.tStart = t  # local t and not account for scr refresh
            B8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(B8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'B8.started')
            # update status
            B8.status = STARTED
            win.callOnFlip(B8.buttonClock.reset)
            B8.setAutoDraw(True)
        
        # if B8 is active this frame...
        if B8.status == STARTED:
            # update params
            pass
            # check whether B8 has been pressed
            if B8.isClicked:
                if not B8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    B8.timesOn.append(B8.buttonClock.getTime())
                    B8.timesOff.append(B8.buttonClock.getTime())
                elif len(B8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    B8.timesOff[-1] = B8.buttonClock.getTime()
                if not B8.wasClicked:
                    # end routine when B8 is clicked
                    continueRoutine = False
                if not B8.wasClicked:
                    # run callback code when B8 is clicked
                    pass
        # take note of whether B8 was clicked, so that next frame we know if clicks are new
        B8.wasClicked = B8.isClicked and B8.status == STARTED
        # *V1* updates
        
        # if V1 is starting this frame...
        if V1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V1.frameNStart = frameN  # exact frame index
            V1.tStart = t  # local t and not account for scr refresh
            V1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V1.started')
            # update status
            V1.status = STARTED
            win.callOnFlip(V1.buttonClock.reset)
            V1.setAutoDraw(True)
        
        # if V1 is active this frame...
        if V1.status == STARTED:
            # update params
            pass
            # check whether V1 has been pressed
            if V1.isClicked:
                if not V1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V1.timesOn.append(V1.buttonClock.getTime())
                    V1.timesOff.append(V1.buttonClock.getTime())
                elif len(V1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V1.timesOff[-1] = V1.buttonClock.getTime()
                if not V1.wasClicked:
                    # end routine when V1 is clicked
                    continueRoutine = False
                if not V1.wasClicked:
                    # run callback code when V1 is clicked
                    pass
        # take note of whether V1 was clicked, so that next frame we know if clicks are new
        V1.wasClicked = V1.isClicked and V1.status == STARTED
        # *V2* updates
        
        # if V2 is starting this frame...
        if V2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V2.frameNStart = frameN  # exact frame index
            V2.tStart = t  # local t and not account for scr refresh
            V2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V2.started')
            # update status
            V2.status = STARTED
            win.callOnFlip(V2.buttonClock.reset)
            V2.setAutoDraw(True)
        
        # if V2 is active this frame...
        if V2.status == STARTED:
            # update params
            pass
            # check whether V2 has been pressed
            if V2.isClicked:
                if not V2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V2.timesOn.append(V2.buttonClock.getTime())
                    V2.timesOff.append(V2.buttonClock.getTime())
                elif len(V2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V2.timesOff[-1] = V2.buttonClock.getTime()
                if not V2.wasClicked:
                    # end routine when V2 is clicked
                    continueRoutine = False
                if not V2.wasClicked:
                    # run callback code when V2 is clicked
                    pass
        # take note of whether V2 was clicked, so that next frame we know if clicks are new
        V2.wasClicked = V2.isClicked and V2.status == STARTED
        # *V3* updates
        
        # if V3 is starting this frame...
        if V3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V3.frameNStart = frameN  # exact frame index
            V3.tStart = t  # local t and not account for scr refresh
            V3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V3.started')
            # update status
            V3.status = STARTED
            win.callOnFlip(V3.buttonClock.reset)
            V3.setAutoDraw(True)
        
        # if V3 is active this frame...
        if V3.status == STARTED:
            # update params
            pass
            # check whether V3 has been pressed
            if V3.isClicked:
                if not V3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V3.timesOn.append(V3.buttonClock.getTime())
                    V3.timesOff.append(V3.buttonClock.getTime())
                elif len(V3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V3.timesOff[-1] = V3.buttonClock.getTime()
                if not V3.wasClicked:
                    # end routine when V3 is clicked
                    continueRoutine = False
                if not V3.wasClicked:
                    # run callback code when V3 is clicked
                    pass
        # take note of whether V3 was clicked, so that next frame we know if clicks are new
        V3.wasClicked = V3.isClicked and V3.status == STARTED
        # *V4* updates
        
        # if V4 is starting this frame...
        if V4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V4.frameNStart = frameN  # exact frame index
            V4.tStart = t  # local t and not account for scr refresh
            V4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V4.started')
            # update status
            V4.status = STARTED
            win.callOnFlip(V4.buttonClock.reset)
            V4.setAutoDraw(True)
        
        # if V4 is active this frame...
        if V4.status == STARTED:
            # update params
            pass
            # check whether V4 has been pressed
            if V4.isClicked:
                if not V4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V4.timesOn.append(V4.buttonClock.getTime())
                    V4.timesOff.append(V4.buttonClock.getTime())
                elif len(V4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V4.timesOff[-1] = V4.buttonClock.getTime()
                if not V4.wasClicked:
                    # end routine when V4 is clicked
                    continueRoutine = False
                if not V4.wasClicked:
                    # run callback code when V4 is clicked
                    pass
        # take note of whether V4 was clicked, so that next frame we know if clicks are new
        V4.wasClicked = V4.isClicked and V4.status == STARTED
        # *V5* updates
        
        # if V5 is starting this frame...
        if V5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V5.frameNStart = frameN  # exact frame index
            V5.tStart = t  # local t and not account for scr refresh
            V5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V5.started')
            # update status
            V5.status = STARTED
            win.callOnFlip(V5.buttonClock.reset)
            V5.setAutoDraw(True)
        
        # if V5 is active this frame...
        if V5.status == STARTED:
            # update params
            pass
            # check whether V5 has been pressed
            if V5.isClicked:
                if not V5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V5.timesOn.append(V5.buttonClock.getTime())
                    V5.timesOff.append(V5.buttonClock.getTime())
                elif len(V5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V5.timesOff[-1] = V5.buttonClock.getTime()
                if not V5.wasClicked:
                    # end routine when V5 is clicked
                    continueRoutine = False
                if not V5.wasClicked:
                    # run callback code when V5 is clicked
                    pass
        # take note of whether V5 was clicked, so that next frame we know if clicks are new
        V5.wasClicked = V5.isClicked and V5.status == STARTED
        # *V6* updates
        
        # if V6 is starting this frame...
        if V6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V6.frameNStart = frameN  # exact frame index
            V6.tStart = t  # local t and not account for scr refresh
            V6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V6.started')
            # update status
            V6.status = STARTED
            win.callOnFlip(V6.buttonClock.reset)
            V6.setAutoDraw(True)
        
        # if V6 is active this frame...
        if V6.status == STARTED:
            # update params
            pass
            # check whether V6 has been pressed
            if V6.isClicked:
                if not V6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V6.timesOn.append(V6.buttonClock.getTime())
                    V6.timesOff.append(V6.buttonClock.getTime())
                elif len(V6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V6.timesOff[-1] = V6.buttonClock.getTime()
                if not V6.wasClicked:
                    # end routine when V6 is clicked
                    continueRoutine = False
                if not V6.wasClicked:
                    # run callback code when V6 is clicked
                    pass
        # take note of whether V6 was clicked, so that next frame we know if clicks are new
        V6.wasClicked = V6.isClicked and V6.status == STARTED
        # *V7* updates
        
        # if V7 is starting this frame...
        if V7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V7.frameNStart = frameN  # exact frame index
            V7.tStart = t  # local t and not account for scr refresh
            V7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V7.started')
            # update status
            V7.status = STARTED
            win.callOnFlip(V7.buttonClock.reset)
            V7.setAutoDraw(True)
        
        # if V7 is active this frame...
        if V7.status == STARTED:
            # update params
            pass
            # check whether V7 has been pressed
            if V7.isClicked:
                if not V7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V7.timesOn.append(V7.buttonClock.getTime())
                    V7.timesOff.append(V7.buttonClock.getTime())
                elif len(V7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V7.timesOff[-1] = V7.buttonClock.getTime()
                if not V7.wasClicked:
                    # end routine when V7 is clicked
                    continueRoutine = False
                if not V7.wasClicked:
                    # run callback code when V7 is clicked
                    pass
        # take note of whether V7 was clicked, so that next frame we know if clicks are new
        V7.wasClicked = V7.isClicked and V7.status == STARTED
        # *V8* updates
        
        # if V8 is starting this frame...
        if V8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            V8.frameNStart = frameN  # exact frame index
            V8.tStart = t  # local t and not account for scr refresh
            V8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(V8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'V8.started')
            # update status
            V8.status = STARTED
            win.callOnFlip(V8.buttonClock.reset)
            V8.setAutoDraw(True)
        
        # if V8 is active this frame...
        if V8.status == STARTED:
            # update params
            pass
            # check whether V8 has been pressed
            if V8.isClicked:
                if not V8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    V8.timesOn.append(V8.buttonClock.getTime())
                    V8.timesOff.append(V8.buttonClock.getTime())
                elif len(V8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    V8.timesOff[-1] = V8.buttonClock.getTime()
                if not V8.wasClicked:
                    # end routine when V8 is clicked
                    continueRoutine = False
                if not V8.wasClicked:
                    # run callback code when V8 is clicked
                    pass
        # take note of whether V8 was clicked, so that next frame we know if clicks are new
        V8.wasClicked = V8.isClicked and V8.status == STARTED
        # *J1* updates
        
        # if J1 is starting this frame...
        if J1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J1.frameNStart = frameN  # exact frame index
            J1.tStart = t  # local t and not account for scr refresh
            J1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J1.started')
            # update status
            J1.status = STARTED
            win.callOnFlip(J1.buttonClock.reset)
            J1.setAutoDraw(True)
        
        # if J1 is active this frame...
        if J1.status == STARTED:
            # update params
            pass
            # check whether J1 has been pressed
            if J1.isClicked:
                if not J1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J1.timesOn.append(J1.buttonClock.getTime())
                    J1.timesOff.append(J1.buttonClock.getTime())
                elif len(J1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J1.timesOff[-1] = J1.buttonClock.getTime()
                if not J1.wasClicked:
                    # end routine when J1 is clicked
                    continueRoutine = False
                if not J1.wasClicked:
                    # run callback code when J1 is clicked
                    pass
        # take note of whether J1 was clicked, so that next frame we know if clicks are new
        J1.wasClicked = J1.isClicked and J1.status == STARTED
        # *J2* updates
        
        # if J2 is starting this frame...
        if J2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J2.frameNStart = frameN  # exact frame index
            J2.tStart = t  # local t and not account for scr refresh
            J2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J2.started')
            # update status
            J2.status = STARTED
            win.callOnFlip(J2.buttonClock.reset)
            J2.setAutoDraw(True)
        
        # if J2 is active this frame...
        if J2.status == STARTED:
            # update params
            pass
            # check whether J2 has been pressed
            if J2.isClicked:
                if not J2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J2.timesOn.append(J2.buttonClock.getTime())
                    J2.timesOff.append(J2.buttonClock.getTime())
                elif len(J2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J2.timesOff[-1] = J2.buttonClock.getTime()
                if not J2.wasClicked:
                    # end routine when J2 is clicked
                    continueRoutine = False
                if not J2.wasClicked:
                    # run callback code when J2 is clicked
                    pass
        # take note of whether J2 was clicked, so that next frame we know if clicks are new
        J2.wasClicked = J2.isClicked and J2.status == STARTED
        # *J3* updates
        
        # if J3 is starting this frame...
        if J3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J3.frameNStart = frameN  # exact frame index
            J3.tStart = t  # local t and not account for scr refresh
            J3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J3.started')
            # update status
            J3.status = STARTED
            win.callOnFlip(J3.buttonClock.reset)
            J3.setAutoDraw(True)
        
        # if J3 is active this frame...
        if J3.status == STARTED:
            # update params
            pass
            # check whether J3 has been pressed
            if J3.isClicked:
                if not J3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J3.timesOn.append(J3.buttonClock.getTime())
                    J3.timesOff.append(J3.buttonClock.getTime())
                elif len(J3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J3.timesOff[-1] = J3.buttonClock.getTime()
                if not J3.wasClicked:
                    # end routine when J3 is clicked
                    continueRoutine = False
                if not J3.wasClicked:
                    # run callback code when J3 is clicked
                    pass
        # take note of whether J3 was clicked, so that next frame we know if clicks are new
        J3.wasClicked = J3.isClicked and J3.status == STARTED
        # *J4* updates
        
        # if J4 is starting this frame...
        if J4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J4.frameNStart = frameN  # exact frame index
            J4.tStart = t  # local t and not account for scr refresh
            J4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J4.started')
            # update status
            J4.status = STARTED
            win.callOnFlip(J4.buttonClock.reset)
            J4.setAutoDraw(True)
        
        # if J4 is active this frame...
        if J4.status == STARTED:
            # update params
            pass
            # check whether J4 has been pressed
            if J4.isClicked:
                if not J4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J4.timesOn.append(J4.buttonClock.getTime())
                    J4.timesOff.append(J4.buttonClock.getTime())
                elif len(J4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J4.timesOff[-1] = J4.buttonClock.getTime()
                if not J4.wasClicked:
                    # end routine when J4 is clicked
                    continueRoutine = False
                if not J4.wasClicked:
                    # run callback code when J4 is clicked
                    pass
        # take note of whether J4 was clicked, so that next frame we know if clicks are new
        J4.wasClicked = J4.isClicked and J4.status == STARTED
        # *J5* updates
        
        # if J5 is starting this frame...
        if J5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J5.frameNStart = frameN  # exact frame index
            J5.tStart = t  # local t and not account for scr refresh
            J5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J5.started')
            # update status
            J5.status = STARTED
            win.callOnFlip(J5.buttonClock.reset)
            J5.setAutoDraw(True)
        
        # if J5 is active this frame...
        if J5.status == STARTED:
            # update params
            pass
            # check whether J5 has been pressed
            if J5.isClicked:
                if not J5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J5.timesOn.append(J5.buttonClock.getTime())
                    J5.timesOff.append(J5.buttonClock.getTime())
                elif len(J5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J5.timesOff[-1] = J5.buttonClock.getTime()
                if not J5.wasClicked:
                    # end routine when J5 is clicked
                    continueRoutine = False
                if not J5.wasClicked:
                    # run callback code when J5 is clicked
                    pass
        # take note of whether J5 was clicked, so that next frame we know if clicks are new
        J5.wasClicked = J5.isClicked and J5.status == STARTED
        # *J6* updates
        
        # if J6 is starting this frame...
        if J6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J6.frameNStart = frameN  # exact frame index
            J6.tStart = t  # local t and not account for scr refresh
            J6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J6.started')
            # update status
            J6.status = STARTED
            win.callOnFlip(J6.buttonClock.reset)
            J6.setAutoDraw(True)
        
        # if J6 is active this frame...
        if J6.status == STARTED:
            # update params
            pass
            # check whether J6 has been pressed
            if J6.isClicked:
                if not J6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J6.timesOn.append(J6.buttonClock.getTime())
                    J6.timesOff.append(J6.buttonClock.getTime())
                elif len(J6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J6.timesOff[-1] = J6.buttonClock.getTime()
                if not J6.wasClicked:
                    # end routine when J6 is clicked
                    continueRoutine = False
                if not J6.wasClicked:
                    # run callback code when J6 is clicked
                    pass
        # take note of whether J6 was clicked, so that next frame we know if clicks are new
        J6.wasClicked = J6.isClicked and J6.status == STARTED
        # *J7* updates
        
        # if J7 is starting this frame...
        if J7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J7.frameNStart = frameN  # exact frame index
            J7.tStart = t  # local t and not account for scr refresh
            J7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J7.started')
            # update status
            J7.status = STARTED
            win.callOnFlip(J7.buttonClock.reset)
            J7.setAutoDraw(True)
        
        # if J7 is active this frame...
        if J7.status == STARTED:
            # update params
            pass
            # check whether J7 has been pressed
            if J7.isClicked:
                if not J7.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J7.timesOn.append(J7.buttonClock.getTime())
                    J7.timesOff.append(J7.buttonClock.getTime())
                elif len(J7.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J7.timesOff[-1] = J7.buttonClock.getTime()
                if not J7.wasClicked:
                    # end routine when J7 is clicked
                    continueRoutine = False
                if not J7.wasClicked:
                    # run callback code when J7 is clicked
                    pass
        # take note of whether J7 was clicked, so that next frame we know if clicks are new
        J7.wasClicked = J7.isClicked and J7.status == STARTED
        # *J8* updates
        
        # if J8 is starting this frame...
        if J8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            J8.frameNStart = frameN  # exact frame index
            J8.tStart = t  # local t and not account for scr refresh
            J8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(J8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'J8.started')
            # update status
            J8.status = STARTED
            win.callOnFlip(J8.buttonClock.reset)
            J8.setAutoDraw(True)
        
        # if J8 is active this frame...
        if J8.status == STARTED:
            # update params
            pass
            # check whether J8 has been pressed
            if J8.isClicked:
                if not J8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    J8.timesOn.append(J8.buttonClock.getTime())
                    J8.timesOff.append(J8.buttonClock.getTime())
                elif len(J8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    J8.timesOff[-1] = J8.buttonClock.getTime()
                if not J8.wasClicked:
                    # end routine when J8 is clicked
                    continueRoutine = False
                if not J8.wasClicked:
                    # run callback code when J8 is clicked
                    pass
        # take note of whether J8 was clicked, so that next frame we know if clicks are new
        J8.wasClicked = J8.isClicked and J8.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=trial,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            trial.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if trial.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in trial.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial" ---
    for thisComponent in trial.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for trial
    trial.tStop = globalClock.getTime(format='float')
    trial.tStopRefresh = tThisFlipGlobal
    thisExp.addData('trial.stopped', trial.tStop)
    thisExp.addData('R1.numClicks', R1.numClicks)
    if R1.numClicks:
       thisExp.addData('R1.timesOn', R1.timesOn)
       thisExp.addData('R1.timesOff', R1.timesOff)
    else:
       thisExp.addData('R1.timesOn', "")
       thisExp.addData('R1.timesOff', "")
    thisExp.addData('R2.numClicks', R2.numClicks)
    if R2.numClicks:
       thisExp.addData('R2.timesOn', R2.timesOn)
       thisExp.addData('R2.timesOff', R2.timesOff)
    else:
       thisExp.addData('R2.timesOn', "")
       thisExp.addData('R2.timesOff', "")
    thisExp.addData('R3.numClicks', R3.numClicks)
    if R3.numClicks:
       thisExp.addData('R3.timesOn', R3.timesOn)
       thisExp.addData('R3.timesOff', R3.timesOff)
    else:
       thisExp.addData('R3.timesOn', "")
       thisExp.addData('R3.timesOff', "")
    thisExp.addData('R4.numClicks', R4.numClicks)
    if R4.numClicks:
       thisExp.addData('R4.timesOn', R4.timesOn)
       thisExp.addData('R4.timesOff', R4.timesOff)
    else:
       thisExp.addData('R4.timesOn', "")
       thisExp.addData('R4.timesOff', "")
    thisExp.addData('R5.numClicks', R5.numClicks)
    if R5.numClicks:
       thisExp.addData('R5.timesOn', R5.timesOn)
       thisExp.addData('R5.timesOff', R5.timesOff)
    else:
       thisExp.addData('R5.timesOn', "")
       thisExp.addData('R5.timesOff', "")
    thisExp.addData('R6.numClicks', R6.numClicks)
    if R6.numClicks:
       thisExp.addData('R6.timesOn', R6.timesOn)
       thisExp.addData('R6.timesOff', R6.timesOff)
    else:
       thisExp.addData('R6.timesOn', "")
       thisExp.addData('R6.timesOff', "")
    thisExp.addData('R7.numClicks', R7.numClicks)
    if R7.numClicks:
       thisExp.addData('R7.timesOn', R7.timesOn)
       thisExp.addData('R7.timesOff', R7.timesOff)
    else:
       thisExp.addData('R7.timesOn', "")
       thisExp.addData('R7.timesOff', "")
    thisExp.addData('R8.numClicks', R8.numClicks)
    if R8.numClicks:
       thisExp.addData('R8.timesOn', R8.timesOn)
       thisExp.addData('R8.timesOff', R8.timesOff)
    else:
       thisExp.addData('R8.timesOn', "")
       thisExp.addData('R8.timesOff', "")
    thisExp.addData('B1.numClicks', B1.numClicks)
    if B1.numClicks:
       thisExp.addData('B1.timesOn', B1.timesOn)
       thisExp.addData('B1.timesOff', B1.timesOff)
    else:
       thisExp.addData('B1.timesOn', "")
       thisExp.addData('B1.timesOff', "")
    thisExp.addData('B2.numClicks', B2.numClicks)
    if B2.numClicks:
       thisExp.addData('B2.timesOn', B2.timesOn)
       thisExp.addData('B2.timesOff', B2.timesOff)
    else:
       thisExp.addData('B2.timesOn', "")
       thisExp.addData('B2.timesOff', "")
    thisExp.addData('B3.numClicks', B3.numClicks)
    if B3.numClicks:
       thisExp.addData('B3.timesOn', B3.timesOn)
       thisExp.addData('B3.timesOff', B3.timesOff)
    else:
       thisExp.addData('B3.timesOn', "")
       thisExp.addData('B3.timesOff', "")
    thisExp.addData('B4.numClicks', B4.numClicks)
    if B4.numClicks:
       thisExp.addData('B4.timesOn', B4.timesOn)
       thisExp.addData('B4.timesOff', B4.timesOff)
    else:
       thisExp.addData('B4.timesOn', "")
       thisExp.addData('B4.timesOff', "")
    thisExp.addData('B5.numClicks', B5.numClicks)
    if B5.numClicks:
       thisExp.addData('B5.timesOn', B5.timesOn)
       thisExp.addData('B5.timesOff', B5.timesOff)
    else:
       thisExp.addData('B5.timesOn', "")
       thisExp.addData('B5.timesOff', "")
    thisExp.addData('B6.numClicks', B6.numClicks)
    if B6.numClicks:
       thisExp.addData('B6.timesOn', B6.timesOn)
       thisExp.addData('B6.timesOff', B6.timesOff)
    else:
       thisExp.addData('B6.timesOn', "")
       thisExp.addData('B6.timesOff', "")
    thisExp.addData('B7.numClicks', B7.numClicks)
    if B7.numClicks:
       thisExp.addData('B7.timesOn', B7.timesOn)
       thisExp.addData('B7.timesOff', B7.timesOff)
    else:
       thisExp.addData('B7.timesOn', "")
       thisExp.addData('B7.timesOff', "")
    thisExp.addData('B8.numClicks', B8.numClicks)
    if B8.numClicks:
       thisExp.addData('B8.timesOn', B8.timesOn)
       thisExp.addData('B8.timesOff', B8.timesOff)
    else:
       thisExp.addData('B8.timesOn', "")
       thisExp.addData('B8.timesOff', "")
    thisExp.addData('V1.numClicks', V1.numClicks)
    if V1.numClicks:
       thisExp.addData('V1.timesOn', V1.timesOn)
       thisExp.addData('V1.timesOff', V1.timesOff)
    else:
       thisExp.addData('V1.timesOn', "")
       thisExp.addData('V1.timesOff', "")
    thisExp.addData('V2.numClicks', V2.numClicks)
    if V2.numClicks:
       thisExp.addData('V2.timesOn', V2.timesOn)
       thisExp.addData('V2.timesOff', V2.timesOff)
    else:
       thisExp.addData('V2.timesOn', "")
       thisExp.addData('V2.timesOff', "")
    thisExp.addData('V3.numClicks', V3.numClicks)
    if V3.numClicks:
       thisExp.addData('V3.timesOn', V3.timesOn)
       thisExp.addData('V3.timesOff', V3.timesOff)
    else:
       thisExp.addData('V3.timesOn', "")
       thisExp.addData('V3.timesOff', "")
    thisExp.addData('V4.numClicks', V4.numClicks)
    if V4.numClicks:
       thisExp.addData('V4.timesOn', V4.timesOn)
       thisExp.addData('V4.timesOff', V4.timesOff)
    else:
       thisExp.addData('V4.timesOn', "")
       thisExp.addData('V4.timesOff', "")
    thisExp.addData('V5.numClicks', V5.numClicks)
    if V5.numClicks:
       thisExp.addData('V5.timesOn', V5.timesOn)
       thisExp.addData('V5.timesOff', V5.timesOff)
    else:
       thisExp.addData('V5.timesOn', "")
       thisExp.addData('V5.timesOff', "")
    thisExp.addData('V6.numClicks', V6.numClicks)
    if V6.numClicks:
       thisExp.addData('V6.timesOn', V6.timesOn)
       thisExp.addData('V6.timesOff', V6.timesOff)
    else:
       thisExp.addData('V6.timesOn', "")
       thisExp.addData('V6.timesOff', "")
    thisExp.addData('V7.numClicks', V7.numClicks)
    if V7.numClicks:
       thisExp.addData('V7.timesOn', V7.timesOn)
       thisExp.addData('V7.timesOff', V7.timesOff)
    else:
       thisExp.addData('V7.timesOn', "")
       thisExp.addData('V7.timesOff', "")
    thisExp.addData('V8.numClicks', V8.numClicks)
    if V8.numClicks:
       thisExp.addData('V8.timesOn', V8.timesOn)
       thisExp.addData('V8.timesOff', V8.timesOff)
    else:
       thisExp.addData('V8.timesOn', "")
       thisExp.addData('V8.timesOff', "")
    thisExp.addData('J1.numClicks', J1.numClicks)
    if J1.numClicks:
       thisExp.addData('J1.timesOn', J1.timesOn)
       thisExp.addData('J1.timesOff', J1.timesOff)
    else:
       thisExp.addData('J1.timesOn', "")
       thisExp.addData('J1.timesOff', "")
    thisExp.addData('J2.numClicks', J2.numClicks)
    if J2.numClicks:
       thisExp.addData('J2.timesOn', J2.timesOn)
       thisExp.addData('J2.timesOff', J2.timesOff)
    else:
       thisExp.addData('J2.timesOn', "")
       thisExp.addData('J2.timesOff', "")
    thisExp.addData('J3.numClicks', J3.numClicks)
    if J3.numClicks:
       thisExp.addData('J3.timesOn', J3.timesOn)
       thisExp.addData('J3.timesOff', J3.timesOff)
    else:
       thisExp.addData('J3.timesOn', "")
       thisExp.addData('J3.timesOff', "")
    thisExp.addData('J4.numClicks', J4.numClicks)
    if J4.numClicks:
       thisExp.addData('J4.timesOn', J4.timesOn)
       thisExp.addData('J4.timesOff', J4.timesOff)
    else:
       thisExp.addData('J4.timesOn', "")
       thisExp.addData('J4.timesOff', "")
    thisExp.addData('J5.numClicks', J5.numClicks)
    if J5.numClicks:
       thisExp.addData('J5.timesOn', J5.timesOn)
       thisExp.addData('J5.timesOff', J5.timesOff)
    else:
       thisExp.addData('J5.timesOn', "")
       thisExp.addData('J5.timesOff', "")
    thisExp.addData('J6.numClicks', J6.numClicks)
    if J6.numClicks:
       thisExp.addData('J6.timesOn', J6.timesOn)
       thisExp.addData('J6.timesOff', J6.timesOff)
    else:
       thisExp.addData('J6.timesOn', "")
       thisExp.addData('J6.timesOff', "")
    thisExp.addData('J7.numClicks', J7.numClicks)
    if J7.numClicks:
       thisExp.addData('J7.timesOn', J7.timesOn)
       thisExp.addData('J7.timesOff', J7.timesOff)
    else:
       thisExp.addData('J7.timesOn', "")
       thisExp.addData('J7.timesOff', "")
    thisExp.addData('J8.numClicks', J8.numClicks)
    if J8.numClicks:
       thisExp.addData('J8.timesOn', J8.timesOn)
       thisExp.addData('J8.timesOff', J8.timesOff)
    else:
       thisExp.addData('J8.timesOn', "")
       thisExp.addData('J8.timesOff', "")
    thisExp.nextEntry()
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "escu" ---
    # create an object to store info about Routine escu
    escu = data.Routine(
        name='escu',
        components=[Pasdeffort, __1, Trespeudeffort],
    )
    escu.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset Pasdeffort to account for continued clicks & clear times on/off
    Pasdeffort.reset()
    # reset __1 to account for continued clicks & clear times on/off
    __1.reset()
    # reset Trespeudeffort to account for continued clicks & clear times on/off
    Trespeudeffort.reset()
    # store start times for escu
    escu.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    escu.tStart = globalClock.getTime(format='float')
    escu.status = STARTED
    thisExp.addData('escu.started', escu.tStart)
    escu.maxDuration = None
    # keep track of which components have finished
    escuComponents = escu.components
    for thisComponent in escu.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "escu" ---
    thisExp.currentRoutine = escu
    escu.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *Pasdeffort* updates
        
        # if Pasdeffort is starting this frame...
        if Pasdeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Pasdeffort.frameNStart = frameN  # exact frame index
            Pasdeffort.tStart = t  # local t and not account for scr refresh
            Pasdeffort.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Pasdeffort, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Pasdeffort.started')
            # update status
            Pasdeffort.status = STARTED
            win.callOnFlip(Pasdeffort.buttonClock.reset)
            Pasdeffort.setAutoDraw(True)
        
        # if Pasdeffort is active this frame...
        if Pasdeffort.status == STARTED:
            # update params
            pass
            # check whether Pasdeffort has been pressed
            if Pasdeffort.isClicked:
                if not Pasdeffort.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    Pasdeffort.timesOn.append(Pasdeffort.buttonClock.getTime())
                    Pasdeffort.timesOff.append(Pasdeffort.buttonClock.getTime())
                elif len(Pasdeffort.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    Pasdeffort.timesOff[-1] = Pasdeffort.buttonClock.getTime()
                if not Pasdeffort.wasClicked:
                    # end routine when Pasdeffort is clicked
                    continueRoutine = False
                if not Pasdeffort.wasClicked:
                    # run callback code when Pasdeffort is clicked
                    pass
        # take note of whether Pasdeffort was clicked, so that next frame we know if clicks are new
        Pasdeffort.wasClicked = Pasdeffort.isClicked and Pasdeffort.status == STARTED
        # *__1* updates
        
        # if __1 is starting this frame...
        if __1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            __1.frameNStart = frameN  # exact frame index
            __1.tStart = t  # local t and not account for scr refresh
            __1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(__1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, '__1.started')
            # update status
            __1.status = STARTED
            win.callOnFlip(__1.buttonClock.reset)
            __1.setAutoDraw(True)
        
        # if __1 is active this frame...
        if __1.status == STARTED:
            # update params
            pass
            # check whether __1 has been pressed
            if __1.isClicked:
                if not __1.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    __1.timesOn.append(__1.buttonClock.getTime())
                    __1.timesOff.append(__1.buttonClock.getTime())
                elif len(__1.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    __1.timesOff[-1] = __1.buttonClock.getTime()
                if not __1.wasClicked:
                    # end routine when __1 is clicked
                    continueRoutine = False
                if not __1.wasClicked:
                    # run callback code when __1 is clicked
                    pass
        # take note of whether __1 was clicked, so that next frame we know if clicks are new
        __1.wasClicked = __1.isClicked and __1.status == STARTED
        # *Trespeudeffort* updates
        
        # if Trespeudeffort is starting this frame...
        if Trespeudeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Trespeudeffort.frameNStart = frameN  # exact frame index
            Trespeudeffort.tStart = t  # local t and not account for scr refresh
            Trespeudeffort.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Trespeudeffort, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Trespeudeffort.started')
            # update status
            Trespeudeffort.status = STARTED
            win.callOnFlip(Trespeudeffort.buttonClock.reset)
            Trespeudeffort.setAutoDraw(True)
        
        # if Trespeudeffort is active this frame...
        if Trespeudeffort.status == STARTED:
            # update params
            pass
            # check whether Trespeudeffort has been pressed
            if Trespeudeffort.isClicked:
                if not Trespeudeffort.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    Trespeudeffort.timesOn.append(Trespeudeffort.buttonClock.getTime())
                    Trespeudeffort.timesOff.append(Trespeudeffort.buttonClock.getTime())
                elif len(Trespeudeffort.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    Trespeudeffort.timesOff[-1] = Trespeudeffort.buttonClock.getTime()
                if not Trespeudeffort.wasClicked:
                    # end routine when Trespeudeffort is clicked
                    continueRoutine = False
                if not Trespeudeffort.wasClicked:
                    # run callback code when Trespeudeffort is clicked
                    pass
        # take note of whether Trespeudeffort was clicked, so that next frame we know if clicks are new
        Trespeudeffort.wasClicked = Trespeudeffort.isClicked and Trespeudeffort.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=escu,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            escu.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if escu.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in escu.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "escu" ---
    for thisComponent in escu.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for escu
    escu.tStop = globalClock.getTime(format='float')
    escu.tStopRefresh = tThisFlipGlobal
    thisExp.addData('escu.stopped', escu.tStop)
    thisExp.addData('Pasdeffort.numClicks', Pasdeffort.numClicks)
    if Pasdeffort.numClicks:
       thisExp.addData('Pasdeffort.timesOn', Pasdeffort.timesOn)
       thisExp.addData('Pasdeffort.timesOff', Pasdeffort.timesOff)
    else:
       thisExp.addData('Pasdeffort.timesOn', "")
       thisExp.addData('Pasdeffort.timesOff', "")
    thisExp.addData('__1.numClicks', __1.numClicks)
    if __1.numClicks:
       thisExp.addData('__1.timesOn', __1.timesOn)
       thisExp.addData('__1.timesOff', __1.timesOff)
    else:
       thisExp.addData('__1.timesOn', "")
       thisExp.addData('__1.timesOff', "")
    thisExp.addData('Trespeudeffort.numClicks', Trespeudeffort.numClicks)
    if Trespeudeffort.numClicks:
       thisExp.addData('Trespeudeffort.timesOn', Trespeudeffort.timesOn)
       thisExp.addData('Trespeudeffort.timesOff', Trespeudeffort.timesOff)
    else:
       thisExp.addData('Trespeudeffort.timesOn', "")
       thisExp.addData('Trespeudeffort.timesOff', "")
    thisExp.nextEntry()
    # the Routine "escu" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
