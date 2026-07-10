#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on juillet 10, 2026, at 15:19
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
        originPath='C:\\Users\\chamery\\Documents\\Spat_CRM_Protocol\\Spat_CRM_Protocol\\src\\CRM_task_lastrun.py',
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
            monitor='testMonitor', color=(1.0000, 1.0000, 1.0000), colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = (1.0000, 1.0000, 1.0000)
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
    # initialize 'Casque_PcHamery'
    deviceManager.addDevice(
        deviceName='Casque_PcHamery',
        index=5.0,
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        resample=True,
        latencyClass=1,
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
    
    # --- Initialize components for Routine "Start" ---
    Starttext = visual.TextStim(win=win, name='Starttext',
        text="SpatFRCMR\n\nÉcoutez les phrases, et reportez la couleur et le chiffre du locuteur qui s'addresse à DELTA.",
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Startbutton = visual.ButtonStim(win, 
        text='Commencer la tâche', font='Arvo',
        pos=(0, -0.15),
        letterHeight=0.05,
        size=(0.5, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Startbutton',
        depth=-1
    )
    Startbutton.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "bloc_2loc" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='Écouter',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
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
    # set audio backend
    sound.Sound.backend = 'ptb'
    Target_1 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='Casque_PcHamery',    name='Target_1'
    )
    Target_1.setVolume(1.0)
    Marsker_1 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='Casque_PcHamery',    name='Marsker_1'
    )
    Marsker_1.setVolume(1.0)
    # Run 'Begin Experiment' code from code
    
    import sofar
    import soundfile as sf
    from scipy.signal import fftconvolve
    from functions import getHRIR
    sofa = sofar.read_sofa("C:\\Users\\chamery\\Documents\\Spat_CRM_Protocol\\Spat_CRM_Protocol\\extfiles\\IRC_1002_R_44100.sofa")
    azimut = [270 ,280 ,290 ,300 ,310 ,320 ,330 ,340 ,350 ,0 , 10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    
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
        pos=(0, -0.33),
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
        pos=(0, -0.26),
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
    __2 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, -0.19),
        letterHeight=0.03,
        size=(0.75, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__2',
        depth=-3
    )
    __2.buttonClock = core.Clock()
    Peudeffort = visual.ButtonStim(win, 
        text="Peu d'effort", font='Arvo',
        pos=(0, -0.12),
        letterHeight=0.03,
        size=(0.7, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Peudeffort',
        depth=-4
    )
    Peudeffort.buttonClock = core.Clock()
    __3 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, -.05),
        letterHeight=0.03,
        size=(0.65, 0.04), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__3',
        depth=-5
    )
    __3.buttonClock = core.Clock()
    Effortmodere = visual.ButtonStim(win, 
        text='Effort modéré', font='Arvo',
        pos=(0, 0.02),
        letterHeight=0.03,
        size=(0.55, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Effortmodere',
        depth=-6
    )
    Effortmodere.buttonClock = core.Clock()
    __4 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, .09),
        letterHeight=0.03,
        size=(0.5, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__4',
        depth=-7
    )
    __4.buttonClock = core.Clock()
    Effortconsiderable = visual.ButtonStim(win, 
        text='Effort considérable', font='Arvo',
        pos=(0, 0.16),
        letterHeight=0.03,
        size=(0.45, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Effortconsiderable',
        depth=-8
    )
    Effortconsiderable.buttonClock = core.Clock()
    __5 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, 0.23),
        letterHeight=0.03,
        size=(0.4, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__5',
        depth=-9
    )
    __5.buttonClock = core.Clock()
    Beaucoupdeffort = visual.ButtonStim(win, 
        text="Beaucoup d'effort", font='Arvo',
        pos=(0, 0.3),
        letterHeight=0.03,
        size=(0.35, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Beaucoupdeffort',
        depth=-10
    )
    Beaucoupdeffort.buttonClock = core.Clock()
    Quedubruit = visual.ButtonStim(win, 
        text='Que du bruit', font='Arvo',
        pos=(0, .4),
        letterHeight=0.03,
        size=(0.3, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Quedubruit',
        depth=-11
    )
    Quedubruit.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "Start_Bloc2" ---
    StartBloc2 = visual.ButtonStim(win, 
        text='Bloc 2 : 3 locuteurs (1 cilbe + 2 mask)', font='Arvo',
        pos=(0, -0.15),
        letterHeight=0.05,
        size=(0.5, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='StartBloc2',
        depth=0
    )
    StartBloc2.buttonClock = core.Clock()
    TextBloc2 = visual.TextStim(win=win, name='TextBloc2',
        text="SpatFRCMR\n\nÉcoutez les phrases, et reportez la couleur et le chiffre du locuteur qui s'addresse à DELTA.",
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "bloc_3loc" ---
    text_instruction_3 = visual.TextStim(win=win, name='text_instruction_3',
        text='Écouter',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    R1_3 = visual.ButtonStim(win, 
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
        name='R1_3',
        depth=-1
    )
    R1_3.buttonClock = core.Clock()
    R2_3 = visual.ButtonStim(win, 
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
        name='R2_3',
        depth=-2
    )
    R2_3.buttonClock = core.Clock()
    R3_3 = visual.ButtonStim(win, 
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
        name='R3_3',
        depth=-3
    )
    R3_3.buttonClock = core.Clock()
    R4_3 = visual.ButtonStim(win, 
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
        name='R4_3',
        depth=-4
    )
    R4_3.buttonClock = core.Clock()
    R5_3 = visual.ButtonStim(win, 
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
        name='R5_3',
        depth=-5
    )
    R5_3.buttonClock = core.Clock()
    R6_3 = visual.ButtonStim(win, 
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
        name='R6_3',
        depth=-6
    )
    R6_3.buttonClock = core.Clock()
    R7_3 = visual.ButtonStim(win, 
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
        name='R7_3',
        depth=-7
    )
    R7_3.buttonClock = core.Clock()
    R8_3 = visual.ButtonStim(win, 
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
        name='R8_3',
        depth=-8
    )
    R8_3.buttonClock = core.Clock()
    B1_3 = visual.ButtonStim(win, 
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
        name='B1_3',
        depth=-9
    )
    B1_3.buttonClock = core.Clock()
    B2_3 = visual.ButtonStim(win, 
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
        name='B2_3',
        depth=-10
    )
    B2_3.buttonClock = core.Clock()
    B3_3 = visual.ButtonStim(win, 
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
        name='B3_3',
        depth=-11
    )
    B3_3.buttonClock = core.Clock()
    B4_3 = visual.ButtonStim(win, 
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
        name='B4_3',
        depth=-12
    )
    B4_3.buttonClock = core.Clock()
    B5_3 = visual.ButtonStim(win, 
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
        name='B5_3',
        depth=-13
    )
    B5_3.buttonClock = core.Clock()
    B6_3 = visual.ButtonStim(win, 
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
        name='B6_3',
        depth=-14
    )
    B6_3.buttonClock = core.Clock()
    B7_3 = visual.ButtonStim(win, 
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
        name='B7_3',
        depth=-15
    )
    B7_3.buttonClock = core.Clock()
    B8_3 = visual.ButtonStim(win, 
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
        name='B8_3',
        depth=-16
    )
    B8_3.buttonClock = core.Clock()
    V1_3 = visual.ButtonStim(win, 
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
        name='V1_3',
        depth=-17
    )
    V1_3.buttonClock = core.Clock()
    V2_3 = visual.ButtonStim(win, 
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
        name='V2_3',
        depth=-18
    )
    V2_3.buttonClock = core.Clock()
    V3_3 = visual.ButtonStim(win, 
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
        name='V3_3',
        depth=-19
    )
    V3_3.buttonClock = core.Clock()
    V4_3 = visual.ButtonStim(win, 
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
        name='V4_3',
        depth=-20
    )
    V4_3.buttonClock = core.Clock()
    V5_3 = visual.ButtonStim(win, 
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
        name='V5_3',
        depth=-21
    )
    V5_3.buttonClock = core.Clock()
    V6_3 = visual.ButtonStim(win, 
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
        name='V6_3',
        depth=-22
    )
    V6_3.buttonClock = core.Clock()
    V7_3 = visual.ButtonStim(win, 
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
        name='V7_3',
        depth=-23
    )
    V7_3.buttonClock = core.Clock()
    V8_3 = visual.ButtonStim(win, 
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
        name='V8_3',
        depth=-24
    )
    V8_3.buttonClock = core.Clock()
    J1_3 = visual.ButtonStim(win, 
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
        name='J1_3',
        depth=-25
    )
    J1_3.buttonClock = core.Clock()
    J2_3 = visual.ButtonStim(win, 
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
        name='J2_3',
        depth=-26
    )
    J2_3.buttonClock = core.Clock()
    J3_3 = visual.ButtonStim(win, 
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
        name='J3_3',
        depth=-27
    )
    J3_3.buttonClock = core.Clock()
    J4_3 = visual.ButtonStim(win, 
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
        name='J4_3',
        depth=-28
    )
    J4_3.buttonClock = core.Clock()
    J5_3 = visual.ButtonStim(win, 
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
        name='J5_3',
        depth=-29
    )
    J5_3.buttonClock = core.Clock()
    J6_3 = visual.ButtonStim(win, 
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
        name='J6_3',
        depth=-30
    )
    J6_3.buttonClock = core.Clock()
    J7_3 = visual.ButtonStim(win, 
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
        name='J7_3',
        depth=-31
    )
    J7_3.buttonClock = core.Clock()
    J8_3 = visual.ButtonStim(win, 
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
        name='J8_3',
        depth=-32
    )
    J8_3.buttonClock = core.Clock()
    Target_2 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='Casque_PcHamery',    name='Target_2'
    )
    Target_2.setVolume(1.0)
    Marsker2_1 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='Casque_PcHamery',    name='Marsker2_1'
    )
    Marsker2_1.setVolume(1.0)
    Masker2_2 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='Casque_PcHamery',    name='Masker2_2'
    )
    Masker2_2.setVolume(1.0)
    
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
        pos=(0, -0.33),
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
        pos=(0, -0.26),
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
    __2 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, -0.19),
        letterHeight=0.03,
        size=(0.75, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__2',
        depth=-3
    )
    __2.buttonClock = core.Clock()
    Peudeffort = visual.ButtonStim(win, 
        text="Peu d'effort", font='Arvo',
        pos=(0, -0.12),
        letterHeight=0.03,
        size=(0.7, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Peudeffort',
        depth=-4
    )
    Peudeffort.buttonClock = core.Clock()
    __3 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, -.05),
        letterHeight=0.03,
        size=(0.65, 0.04), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__3',
        depth=-5
    )
    __3.buttonClock = core.Clock()
    Effortmodere = visual.ButtonStim(win, 
        text='Effort modéré', font='Arvo',
        pos=(0, 0.02),
        letterHeight=0.03,
        size=(0.55, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Effortmodere',
        depth=-6
    )
    Effortmodere.buttonClock = core.Clock()
    __4 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, .09),
        letterHeight=0.03,
        size=(0.5, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__4',
        depth=-7
    )
    __4.buttonClock = core.Clock()
    Effortconsiderable = visual.ButtonStim(win, 
        text='Effort considérable', font='Arvo',
        pos=(0, 0.16),
        letterHeight=0.03,
        size=(0.45, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Effortconsiderable',
        depth=-8
    )
    Effortconsiderable.buttonClock = core.Clock()
    __5 = visual.ButtonStim(win, 
        text='===', font='Arvo',
        pos=(0, 0.23),
        letterHeight=0.03,
        size=(0.4, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='__5',
        depth=-9
    )
    __5.buttonClock = core.Clock()
    Beaucoupdeffort = visual.ButtonStim(win, 
        text="Beaucoup d'effort", font='Arvo',
        pos=(0, 0.3),
        letterHeight=0.03,
        size=(0.35, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Beaucoupdeffort',
        depth=-10
    )
    Beaucoupdeffort.buttonClock = core.Clock()
    Quedubruit = visual.ButtonStim(win, 
        text='Que du bruit', font='Arvo',
        pos=(0, .4),
        letterHeight=0.03,
        size=(0.3, 0.05), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='Quedubruit',
        depth=-11
    )
    Quedubruit.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "End" ---
    EndButton = visual.ButtonStim(win, 
        text='Quitter', font='Arvo',
        pos=(0, -0.15),
        letterHeight=0.05,
        size=(0.5, 0.2), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=(0.3804, 0.5373, 0.7412), borderColor=None,
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='EndButton',
        depth=0
    )
    EndButton.buttonClock = core.Clock()
    Endtext = visual.TextStim(win=win, name='Endtext',
        text='Tâche terminée!\n\nMerci!',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
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
    
    # --- Prepare to start Routine "Start" ---
    # create an object to store info about Routine Start
    Start = data.Routine(
        name='Start',
        components=[Starttext, Startbutton],
    )
    Start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset Startbutton to account for continued clicks & clear times on/off
    Startbutton.reset()
    # store start times for Start
    Start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start.tStart = globalClock.getTime(format='float')
    Start.status = STARTED
    thisExp.addData('Start.started', Start.tStart)
    Start.maxDuration = None
    # keep track of which components have finished
    StartComponents = Start.components
    for thisComponent in Start.components:
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
    
    # --- Run Routine "Start" ---
    thisExp.currentRoutine = Start
    Start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Starttext* updates
        
        # if Starttext is starting this frame...
        if Starttext.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Starttext.frameNStart = frameN  # exact frame index
            Starttext.tStart = t  # local t and not account for scr refresh
            Starttext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Starttext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Starttext.started')
            # update status
            Starttext.status = STARTED
            Starttext.setAutoDraw(True)
        
        # if Starttext is active this frame...
        if Starttext.status == STARTED:
            # update params
            pass
        # *Startbutton* updates
        
        # if Startbutton is starting this frame...
        if Startbutton.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Startbutton.frameNStart = frameN  # exact frame index
            Startbutton.tStart = t  # local t and not account for scr refresh
            Startbutton.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Startbutton, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Startbutton.started')
            # update status
            Startbutton.status = STARTED
            win.callOnFlip(Startbutton.buttonClock.reset)
            Startbutton.setAutoDraw(True)
        
        # if Startbutton is active this frame...
        if Startbutton.status == STARTED:
            # update params
            pass
            # check whether Startbutton has been pressed
            if Startbutton.isClicked:
                if not Startbutton.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    Startbutton.timesOn.append(Startbutton.buttonClock.getTime())
                    Startbutton.timesOff.append(Startbutton.buttonClock.getTime())
                elif len(Startbutton.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    Startbutton.timesOff[-1] = Startbutton.buttonClock.getTime()
                if not Startbutton.wasClicked:
                    # end routine when Startbutton is clicked
                    continueRoutine = False
                if not Startbutton.wasClicked:
                    # run callback code when Startbutton is clicked
                    pass
        # take note of whether Startbutton was clicked, so that next frame we know if clicks are new
        Startbutton.wasClicked = Startbutton.isClicked and Startbutton.status == STARTED
        
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
                currentRoutine=Start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start" ---
    for thisComponent in Start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start
    Start.tStop = globalClock.getTime(format='float')
    Start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start.stopped', Start.tStop)
    thisExp.addData('Startbutton.numClicks', Startbutton.numClicks)
    if Startbutton.numClicks:
       thisExp.addData('Startbutton.timesOn', Startbutton.timesOn)
       thisExp.addData('Startbutton.timesOff', Startbutton.timesOff)
    else:
       thisExp.addData('Startbutton.timesOn', "")
       thisExp.addData('Startbutton.timesOff', "")
    thisExp.nextEntry()
    # the Routine "Start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2loc = data.TrialHandler2(
        name='trials_2loc',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions/conditions_2loc.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials_2loc)  # add the loop to the experiment
    thisTrials_2loc = trials_2loc.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_2loc.rgb)
    if thisTrials_2loc != None:
        for paramName in thisTrials_2loc:
            globals()[paramName] = thisTrials_2loc[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_2loc in trials_2loc:
        trials_2loc.status = STARTED
        if hasattr(thisTrials_2loc, 'status'):
            thisTrials_2loc.status = STARTED
        currentLoop = trials_2loc
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_2loc.rgb)
        if thisTrials_2loc != None:
            for paramName in thisTrials_2loc:
                globals()[paramName] = thisTrials_2loc[paramName]
        
        # --- Prepare to start Routine "bloc_2loc" ---
        # create an object to store info about Routine bloc_2loc
        bloc_2loc = data.Routine(
            name='bloc_2loc',
            components=[text_instruction, R1, R2, R3, R4, R5, R6, R7, R8, B1, B2, B3, B4, B5, B6, B7, B8, V1, V2, V3, V4, V5, V6, V7, V8, J1, J2, J3, J4, J5, J6, J7, J8, Target_1, Marsker_1],
        )
        bloc_2loc.status = NOT_STARTED
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
        Target_1.setSound(Target_2loc, hamming=True)
        Target_1.setVolume(1.0, log=False)
        Target_1.seek(0)
        Marsker_1.setSound(Mask_2loc, hamming=True)
        Marsker_1.setVolume(1.0, log=False)
        Marsker_1.seek(0)
        # Run 'Begin Routine' code from code
        az = 90
        ###hrtf
        hrir_left, hrir_right = getHRIR(az, 0) # az : angle azimuth, 0 pas d'élévation
        
        audio, sr = sf.read(Target_2loc)
        
        left = fftconvolve(audio, hrir_left)
        right = fftconvolve(audio, hrir_right)
        stereo = np.column_stack((left, right))
        stereo /= np.max(np.abs(stereo))
        stereo = stereo.astype(np.float32) #pour ptb
        Target_1 = sound.Sound(
        stereo,
        sampleRate=sr,
        stereo=True,
        hamming=False,     
        speaker='Casque_PcHamery',
        name='Target_1_binaural'
        )
        Target_1.setVolume(1.0, log=False)
        # store start times for bloc_2loc
        bloc_2loc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        bloc_2loc.tStart = globalClock.getTime(format='float')
        bloc_2loc.status = STARTED
        thisExp.addData('bloc_2loc.started', bloc_2loc.tStart)
        bloc_2loc.maxDuration = None
        # keep track of which components have finished
        bloc_2locComponents = bloc_2loc.components
        for thisComponent in bloc_2loc.components:
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
        
        # --- Run Routine "bloc_2loc" ---
        thisExp.currentRoutine = bloc_2loc
        bloc_2loc.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_2loc, 'status') and thisTrials_2loc.status == STOPPING:
                continueRoutine = False
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
                if tThisFlipGlobal > text_instruction.tStartRefresh + 0.5-frameTolerance:
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
            if R1.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if R2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if R3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if R4.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if R5.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if R6.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if R7.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if R8.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if B1.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if B2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if B3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if B4.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if B5.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if B6.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if B7.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if B8.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if V1.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if V2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if V3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if V4.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if V5.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if V6.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if V7.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if V8.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if J1.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if J2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if J3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if J4.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if J5.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if J6.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if J7.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            if J8.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
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
            
            # *Target_1* updates
            
            # if Target_1 is starting this frame...
            if Target_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                Target_1.frameNStart = frameN  # exact frame index
                Target_1.tStart = t  # local t and not account for scr refresh
                Target_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('Target_1.started', tThisFlipGlobal)
                # update status
                Target_1.status = STARTED
                Target_1.play(when=win)  # sync with win flip
            
            # if Target_1 is stopping this frame...
            if Target_1.status == STARTED:
                if bool(False) or Target_1.isFinished:
                    # keep track of stop time/frame for later
                    Target_1.tStop = t  # not accounting for scr refresh
                    Target_1.tStopRefresh = tThisFlipGlobal  # on global time
                    Target_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Target_1.stopped')
                    # update status
                    Target_1.status = FINISHED
                    Target_1.stop()
            
            # *Marsker_1* updates
            
            # if Marsker_1 is starting this frame...
            if Marsker_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                Marsker_1.frameNStart = frameN  # exact frame index
                Marsker_1.tStart = t  # local t and not account for scr refresh
                Marsker_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('Marsker_1.started', tThisFlipGlobal)
                # update status
                Marsker_1.status = STARTED
                Marsker_1.play(when=win)  # sync with win flip
            
            # if Marsker_1 is stopping this frame...
            if Marsker_1.status == STARTED:
                if bool(False) or Marsker_1.isFinished:
                    # keep track of stop time/frame for later
                    Marsker_1.tStop = t  # not accounting for scr refresh
                    Marsker_1.tStopRefresh = tThisFlipGlobal  # on global time
                    Marsker_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Marsker_1.stopped')
                    # update status
                    Marsker_1.status = FINISHED
                    Marsker_1.stop()
            # Run 'Each Frame' code from code
            az = 90
            ###hrtf
            hrir_left, hrir_right = getHRIR(az, 0) # az : angle azimuth, 0 pas d'élévation
            
            audio, sr = sf.read(Target_2loc)
            
            left = fftconvolve(audio, hrir_left)
            right = fftconvolve(audio, hrir_right)
            stereo = np.column_stack((left, right))
            stereo /= np.max(np.abs(stereo))
            stereo = stereo.astype(np.float32) #pour ptb
            Target_1 = sound.Sound(
            stereo,
            sampleRate=sr,
            stereo=True,
            hamming=False,     
            speaker='Casque_PcHamery',
            name='Target_1_binaural'
            )
            Target_1.setVolume(1.0, log=False)
            
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
                    currentRoutine=bloc_2loc,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                bloc_2loc.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if bloc_2loc.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in bloc_2loc.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "bloc_2loc" ---
        for thisComponent in bloc_2loc.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for bloc_2loc
        bloc_2loc.tStop = globalClock.getTime(format='float')
        bloc_2loc.tStopRefresh = tThisFlipGlobal
        thisExp.addData('bloc_2loc.stopped', bloc_2loc.tStop)
        trials_2loc.addData('R1.numClicks', R1.numClicks)
        if R1.numClicks:
           trials_2loc.addData('R1.timesOn', R1.timesOn)
           trials_2loc.addData('R1.timesOff', R1.timesOff)
        else:
           trials_2loc.addData('R1.timesOn', "")
           trials_2loc.addData('R1.timesOff', "")
        trials_2loc.addData('R2.numClicks', R2.numClicks)
        if R2.numClicks:
           trials_2loc.addData('R2.timesOn', R2.timesOn)
           trials_2loc.addData('R2.timesOff', R2.timesOff)
        else:
           trials_2loc.addData('R2.timesOn', "")
           trials_2loc.addData('R2.timesOff', "")
        trials_2loc.addData('R3.numClicks', R3.numClicks)
        if R3.numClicks:
           trials_2loc.addData('R3.timesOn', R3.timesOn)
           trials_2loc.addData('R3.timesOff', R3.timesOff)
        else:
           trials_2loc.addData('R3.timesOn', "")
           trials_2loc.addData('R3.timesOff', "")
        trials_2loc.addData('R4.numClicks', R4.numClicks)
        if R4.numClicks:
           trials_2loc.addData('R4.timesOn', R4.timesOn)
           trials_2loc.addData('R4.timesOff', R4.timesOff)
        else:
           trials_2loc.addData('R4.timesOn', "")
           trials_2loc.addData('R4.timesOff', "")
        trials_2loc.addData('R5.numClicks', R5.numClicks)
        if R5.numClicks:
           trials_2loc.addData('R5.timesOn', R5.timesOn)
           trials_2loc.addData('R5.timesOff', R5.timesOff)
        else:
           trials_2loc.addData('R5.timesOn', "")
           trials_2loc.addData('R5.timesOff', "")
        trials_2loc.addData('R6.numClicks', R6.numClicks)
        if R6.numClicks:
           trials_2loc.addData('R6.timesOn', R6.timesOn)
           trials_2loc.addData('R6.timesOff', R6.timesOff)
        else:
           trials_2loc.addData('R6.timesOn', "")
           trials_2loc.addData('R6.timesOff', "")
        trials_2loc.addData('R7.numClicks', R7.numClicks)
        if R7.numClicks:
           trials_2loc.addData('R7.timesOn', R7.timesOn)
           trials_2loc.addData('R7.timesOff', R7.timesOff)
        else:
           trials_2loc.addData('R7.timesOn', "")
           trials_2loc.addData('R7.timesOff', "")
        trials_2loc.addData('R8.numClicks', R8.numClicks)
        if R8.numClicks:
           trials_2loc.addData('R8.timesOn', R8.timesOn)
           trials_2loc.addData('R8.timesOff', R8.timesOff)
        else:
           trials_2loc.addData('R8.timesOn', "")
           trials_2loc.addData('R8.timesOff', "")
        trials_2loc.addData('B1.numClicks', B1.numClicks)
        if B1.numClicks:
           trials_2loc.addData('B1.timesOn', B1.timesOn)
           trials_2loc.addData('B1.timesOff', B1.timesOff)
        else:
           trials_2loc.addData('B1.timesOn', "")
           trials_2loc.addData('B1.timesOff', "")
        trials_2loc.addData('B2.numClicks', B2.numClicks)
        if B2.numClicks:
           trials_2loc.addData('B2.timesOn', B2.timesOn)
           trials_2loc.addData('B2.timesOff', B2.timesOff)
        else:
           trials_2loc.addData('B2.timesOn', "")
           trials_2loc.addData('B2.timesOff', "")
        trials_2loc.addData('B3.numClicks', B3.numClicks)
        if B3.numClicks:
           trials_2loc.addData('B3.timesOn', B3.timesOn)
           trials_2loc.addData('B3.timesOff', B3.timesOff)
        else:
           trials_2loc.addData('B3.timesOn', "")
           trials_2loc.addData('B3.timesOff', "")
        trials_2loc.addData('B4.numClicks', B4.numClicks)
        if B4.numClicks:
           trials_2loc.addData('B4.timesOn', B4.timesOn)
           trials_2loc.addData('B4.timesOff', B4.timesOff)
        else:
           trials_2loc.addData('B4.timesOn', "")
           trials_2loc.addData('B4.timesOff', "")
        trials_2loc.addData('B5.numClicks', B5.numClicks)
        if B5.numClicks:
           trials_2loc.addData('B5.timesOn', B5.timesOn)
           trials_2loc.addData('B5.timesOff', B5.timesOff)
        else:
           trials_2loc.addData('B5.timesOn', "")
           trials_2loc.addData('B5.timesOff', "")
        trials_2loc.addData('B6.numClicks', B6.numClicks)
        if B6.numClicks:
           trials_2loc.addData('B6.timesOn', B6.timesOn)
           trials_2loc.addData('B6.timesOff', B6.timesOff)
        else:
           trials_2loc.addData('B6.timesOn', "")
           trials_2loc.addData('B6.timesOff', "")
        trials_2loc.addData('B7.numClicks', B7.numClicks)
        if B7.numClicks:
           trials_2loc.addData('B7.timesOn', B7.timesOn)
           trials_2loc.addData('B7.timesOff', B7.timesOff)
        else:
           trials_2loc.addData('B7.timesOn', "")
           trials_2loc.addData('B7.timesOff', "")
        trials_2loc.addData('B8.numClicks', B8.numClicks)
        if B8.numClicks:
           trials_2loc.addData('B8.timesOn', B8.timesOn)
           trials_2loc.addData('B8.timesOff', B8.timesOff)
        else:
           trials_2loc.addData('B8.timesOn', "")
           trials_2loc.addData('B8.timesOff', "")
        trials_2loc.addData('V1.numClicks', V1.numClicks)
        if V1.numClicks:
           trials_2loc.addData('V1.timesOn', V1.timesOn)
           trials_2loc.addData('V1.timesOff', V1.timesOff)
        else:
           trials_2loc.addData('V1.timesOn', "")
           trials_2loc.addData('V1.timesOff', "")
        trials_2loc.addData('V2.numClicks', V2.numClicks)
        if V2.numClicks:
           trials_2loc.addData('V2.timesOn', V2.timesOn)
           trials_2loc.addData('V2.timesOff', V2.timesOff)
        else:
           trials_2loc.addData('V2.timesOn', "")
           trials_2loc.addData('V2.timesOff', "")
        trials_2loc.addData('V3.numClicks', V3.numClicks)
        if V3.numClicks:
           trials_2loc.addData('V3.timesOn', V3.timesOn)
           trials_2loc.addData('V3.timesOff', V3.timesOff)
        else:
           trials_2loc.addData('V3.timesOn', "")
           trials_2loc.addData('V3.timesOff', "")
        trials_2loc.addData('V4.numClicks', V4.numClicks)
        if V4.numClicks:
           trials_2loc.addData('V4.timesOn', V4.timesOn)
           trials_2loc.addData('V4.timesOff', V4.timesOff)
        else:
           trials_2loc.addData('V4.timesOn', "")
           trials_2loc.addData('V4.timesOff', "")
        trials_2loc.addData('V5.numClicks', V5.numClicks)
        if V5.numClicks:
           trials_2loc.addData('V5.timesOn', V5.timesOn)
           trials_2loc.addData('V5.timesOff', V5.timesOff)
        else:
           trials_2loc.addData('V5.timesOn', "")
           trials_2loc.addData('V5.timesOff', "")
        trials_2loc.addData('V6.numClicks', V6.numClicks)
        if V6.numClicks:
           trials_2loc.addData('V6.timesOn', V6.timesOn)
           trials_2loc.addData('V6.timesOff', V6.timesOff)
        else:
           trials_2loc.addData('V6.timesOn', "")
           trials_2loc.addData('V6.timesOff', "")
        trials_2loc.addData('V7.numClicks', V7.numClicks)
        if V7.numClicks:
           trials_2loc.addData('V7.timesOn', V7.timesOn)
           trials_2loc.addData('V7.timesOff', V7.timesOff)
        else:
           trials_2loc.addData('V7.timesOn', "")
           trials_2loc.addData('V7.timesOff', "")
        trials_2loc.addData('V8.numClicks', V8.numClicks)
        if V8.numClicks:
           trials_2loc.addData('V8.timesOn', V8.timesOn)
           trials_2loc.addData('V8.timesOff', V8.timesOff)
        else:
           trials_2loc.addData('V8.timesOn', "")
           trials_2loc.addData('V8.timesOff', "")
        trials_2loc.addData('J1.numClicks', J1.numClicks)
        if J1.numClicks:
           trials_2loc.addData('J1.timesOn', J1.timesOn)
           trials_2loc.addData('J1.timesOff', J1.timesOff)
        else:
           trials_2loc.addData('J1.timesOn', "")
           trials_2loc.addData('J1.timesOff', "")
        trials_2loc.addData('J2.numClicks', J2.numClicks)
        if J2.numClicks:
           trials_2loc.addData('J2.timesOn', J2.timesOn)
           trials_2loc.addData('J2.timesOff', J2.timesOff)
        else:
           trials_2loc.addData('J2.timesOn', "")
           trials_2loc.addData('J2.timesOff', "")
        trials_2loc.addData('J3.numClicks', J3.numClicks)
        if J3.numClicks:
           trials_2loc.addData('J3.timesOn', J3.timesOn)
           trials_2loc.addData('J3.timesOff', J3.timesOff)
        else:
           trials_2loc.addData('J3.timesOn', "")
           trials_2loc.addData('J3.timesOff', "")
        trials_2loc.addData('J4.numClicks', J4.numClicks)
        if J4.numClicks:
           trials_2loc.addData('J4.timesOn', J4.timesOn)
           trials_2loc.addData('J4.timesOff', J4.timesOff)
        else:
           trials_2loc.addData('J4.timesOn', "")
           trials_2loc.addData('J4.timesOff', "")
        trials_2loc.addData('J5.numClicks', J5.numClicks)
        if J5.numClicks:
           trials_2loc.addData('J5.timesOn', J5.timesOn)
           trials_2loc.addData('J5.timesOff', J5.timesOff)
        else:
           trials_2loc.addData('J5.timesOn', "")
           trials_2loc.addData('J5.timesOff', "")
        trials_2loc.addData('J6.numClicks', J6.numClicks)
        if J6.numClicks:
           trials_2loc.addData('J6.timesOn', J6.timesOn)
           trials_2loc.addData('J6.timesOff', J6.timesOff)
        else:
           trials_2loc.addData('J6.timesOn', "")
           trials_2loc.addData('J6.timesOff', "")
        trials_2loc.addData('J7.numClicks', J7.numClicks)
        if J7.numClicks:
           trials_2loc.addData('J7.timesOn', J7.timesOn)
           trials_2loc.addData('J7.timesOff', J7.timesOff)
        else:
           trials_2loc.addData('J7.timesOn', "")
           trials_2loc.addData('J7.timesOff', "")
        trials_2loc.addData('J8.numClicks', J8.numClicks)
        if J8.numClicks:
           trials_2loc.addData('J8.timesOn', J8.timesOn)
           trials_2loc.addData('J8.timesOff', J8.timesOff)
        else:
           trials_2loc.addData('J8.timesOn', "")
           trials_2loc.addData('J8.timesOff', "")
        Target_1.pause()  # ensure sound has stopped at end of Routine
        Marsker_1.pause()  # ensure sound has stopped at end of Routine
        # the Routine "bloc_2loc" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "escu" ---
        # create an object to store info about Routine escu
        escu = data.Routine(
            name='escu',
            components=[Pasdeffort, __1, Trespeudeffort, __2, Peudeffort, __3, Effortmodere, __4, Effortconsiderable, __5, Beaucoupdeffort, Quedubruit],
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
        # reset __2 to account for continued clicks & clear times on/off
        __2.reset()
        # reset Peudeffort to account for continued clicks & clear times on/off
        Peudeffort.reset()
        # reset __3 to account for continued clicks & clear times on/off
        __3.reset()
        # reset Effortmodere to account for continued clicks & clear times on/off
        Effortmodere.reset()
        # reset __4 to account for continued clicks & clear times on/off
        __4.reset()
        # reset Effortconsiderable to account for continued clicks & clear times on/off
        Effortconsiderable.reset()
        # reset __5 to account for continued clicks & clear times on/off
        __5.reset()
        # reset Beaucoupdeffort to account for continued clicks & clear times on/off
        Beaucoupdeffort.reset()
        # reset Quedubruit to account for continued clicks & clear times on/off
        Quedubruit.reset()
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
            # if trial has changed, end Routine now
            if hasattr(thisTrials_2loc, 'status') and thisTrials_2loc.status == STOPPING:
                continueRoutine = False
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
            # *__2* updates
            
            # if __2 is starting this frame...
            if __2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                __2.frameNStart = frameN  # exact frame index
                __2.tStart = t  # local t and not account for scr refresh
                __2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(__2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, '__2.started')
                # update status
                __2.status = STARTED
                win.callOnFlip(__2.buttonClock.reset)
                __2.setAutoDraw(True)
            
            # if __2 is active this frame...
            if __2.status == STARTED:
                # update params
                pass
                # check whether __2 has been pressed
                if __2.isClicked:
                    if not __2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        __2.timesOn.append(__2.buttonClock.getTime())
                        __2.timesOff.append(__2.buttonClock.getTime())
                    elif len(__2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        __2.timesOff[-1] = __2.buttonClock.getTime()
                    if not __2.wasClicked:
                        # end routine when __2 is clicked
                        continueRoutine = False
                    if not __2.wasClicked:
                        # run callback code when __2 is clicked
                        pass
            # take note of whether __2 was clicked, so that next frame we know if clicks are new
            __2.wasClicked = __2.isClicked and __2.status == STARTED
            # *Peudeffort* updates
            
            # if Peudeffort is starting this frame...
            if Peudeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Peudeffort.frameNStart = frameN  # exact frame index
                Peudeffort.tStart = t  # local t and not account for scr refresh
                Peudeffort.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Peudeffort, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Peudeffort.started')
                # update status
                Peudeffort.status = STARTED
                win.callOnFlip(Peudeffort.buttonClock.reset)
                Peudeffort.setAutoDraw(True)
            
            # if Peudeffort is active this frame...
            if Peudeffort.status == STARTED:
                # update params
                pass
                # check whether Peudeffort has been pressed
                if Peudeffort.isClicked:
                    if not Peudeffort.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Peudeffort.timesOn.append(Peudeffort.buttonClock.getTime())
                        Peudeffort.timesOff.append(Peudeffort.buttonClock.getTime())
                    elif len(Peudeffort.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Peudeffort.timesOff[-1] = Peudeffort.buttonClock.getTime()
                    if not Peudeffort.wasClicked:
                        # end routine when Peudeffort is clicked
                        continueRoutine = False
                    if not Peudeffort.wasClicked:
                        # run callback code when Peudeffort is clicked
                        pass
            # take note of whether Peudeffort was clicked, so that next frame we know if clicks are new
            Peudeffort.wasClicked = Peudeffort.isClicked and Peudeffort.status == STARTED
            # *__3* updates
            
            # if __3 is starting this frame...
            if __3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                __3.frameNStart = frameN  # exact frame index
                __3.tStart = t  # local t and not account for scr refresh
                __3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(__3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, '__3.started')
                # update status
                __3.status = STARTED
                win.callOnFlip(__3.buttonClock.reset)
                __3.setAutoDraw(True)
            
            # if __3 is active this frame...
            if __3.status == STARTED:
                # update params
                pass
                # check whether __3 has been pressed
                if __3.isClicked:
                    if not __3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        __3.timesOn.append(__3.buttonClock.getTime())
                        __3.timesOff.append(__3.buttonClock.getTime())
                    elif len(__3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        __3.timesOff[-1] = __3.buttonClock.getTime()
                    if not __3.wasClicked:
                        # end routine when __3 is clicked
                        continueRoutine = False
                    if not __3.wasClicked:
                        # run callback code when __3 is clicked
                        pass
            # take note of whether __3 was clicked, so that next frame we know if clicks are new
            __3.wasClicked = __3.isClicked and __3.status == STARTED
            # *Effortmodere* updates
            
            # if Effortmodere is starting this frame...
            if Effortmodere.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Effortmodere.frameNStart = frameN  # exact frame index
                Effortmodere.tStart = t  # local t and not account for scr refresh
                Effortmodere.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Effortmodere, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Effortmodere.started')
                # update status
                Effortmodere.status = STARTED
                win.callOnFlip(Effortmodere.buttonClock.reset)
                Effortmodere.setAutoDraw(True)
            
            # if Effortmodere is active this frame...
            if Effortmodere.status == STARTED:
                # update params
                pass
                # check whether Effortmodere has been pressed
                if Effortmodere.isClicked:
                    if not Effortmodere.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Effortmodere.timesOn.append(Effortmodere.buttonClock.getTime())
                        Effortmodere.timesOff.append(Effortmodere.buttonClock.getTime())
                    elif len(Effortmodere.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Effortmodere.timesOff[-1] = Effortmodere.buttonClock.getTime()
                    if not Effortmodere.wasClicked:
                        # end routine when Effortmodere is clicked
                        continueRoutine = False
                    if not Effortmodere.wasClicked:
                        # run callback code when Effortmodere is clicked
                        pass
            # take note of whether Effortmodere was clicked, so that next frame we know if clicks are new
            Effortmodere.wasClicked = Effortmodere.isClicked and Effortmodere.status == STARTED
            # *__4* updates
            
            # if __4 is starting this frame...
            if __4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                __4.frameNStart = frameN  # exact frame index
                __4.tStart = t  # local t and not account for scr refresh
                __4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(__4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, '__4.started')
                # update status
                __4.status = STARTED
                win.callOnFlip(__4.buttonClock.reset)
                __4.setAutoDraw(True)
            
            # if __4 is active this frame...
            if __4.status == STARTED:
                # update params
                pass
                # check whether __4 has been pressed
                if __4.isClicked:
                    if not __4.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        __4.timesOn.append(__4.buttonClock.getTime())
                        __4.timesOff.append(__4.buttonClock.getTime())
                    elif len(__4.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        __4.timesOff[-1] = __4.buttonClock.getTime()
                    if not __4.wasClicked:
                        # end routine when __4 is clicked
                        continueRoutine = False
                    if not __4.wasClicked:
                        # run callback code when __4 is clicked
                        pass
            # take note of whether __4 was clicked, so that next frame we know if clicks are new
            __4.wasClicked = __4.isClicked and __4.status == STARTED
            # *Effortconsiderable* updates
            
            # if Effortconsiderable is starting this frame...
            if Effortconsiderable.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Effortconsiderable.frameNStart = frameN  # exact frame index
                Effortconsiderable.tStart = t  # local t and not account for scr refresh
                Effortconsiderable.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Effortconsiderable, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Effortconsiderable.started')
                # update status
                Effortconsiderable.status = STARTED
                win.callOnFlip(Effortconsiderable.buttonClock.reset)
                Effortconsiderable.setAutoDraw(True)
            
            # if Effortconsiderable is active this frame...
            if Effortconsiderable.status == STARTED:
                # update params
                pass
                # check whether Effortconsiderable has been pressed
                if Effortconsiderable.isClicked:
                    if not Effortconsiderable.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Effortconsiderable.timesOn.append(Effortconsiderable.buttonClock.getTime())
                        Effortconsiderable.timesOff.append(Effortconsiderable.buttonClock.getTime())
                    elif len(Effortconsiderable.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Effortconsiderable.timesOff[-1] = Effortconsiderable.buttonClock.getTime()
                    if not Effortconsiderable.wasClicked:
                        # end routine when Effortconsiderable is clicked
                        continueRoutine = False
                    if not Effortconsiderable.wasClicked:
                        # run callback code when Effortconsiderable is clicked
                        pass
            # take note of whether Effortconsiderable was clicked, so that next frame we know if clicks are new
            Effortconsiderable.wasClicked = Effortconsiderable.isClicked and Effortconsiderable.status == STARTED
            # *__5* updates
            
            # if __5 is starting this frame...
            if __5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                __5.frameNStart = frameN  # exact frame index
                __5.tStart = t  # local t and not account for scr refresh
                __5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(__5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, '__5.started')
                # update status
                __5.status = STARTED
                win.callOnFlip(__5.buttonClock.reset)
                __5.setAutoDraw(True)
            
            # if __5 is active this frame...
            if __5.status == STARTED:
                # update params
                pass
                # check whether __5 has been pressed
                if __5.isClicked:
                    if not __5.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        __5.timesOn.append(__5.buttonClock.getTime())
                        __5.timesOff.append(__5.buttonClock.getTime())
                    elif len(__5.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        __5.timesOff[-1] = __5.buttonClock.getTime()
                    if not __5.wasClicked:
                        # end routine when __5 is clicked
                        continueRoutine = False
                    if not __5.wasClicked:
                        # run callback code when __5 is clicked
                        pass
            # take note of whether __5 was clicked, so that next frame we know if clicks are new
            __5.wasClicked = __5.isClicked and __5.status == STARTED
            # *Beaucoupdeffort* updates
            
            # if Beaucoupdeffort is starting this frame...
            if Beaucoupdeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Beaucoupdeffort.frameNStart = frameN  # exact frame index
                Beaucoupdeffort.tStart = t  # local t and not account for scr refresh
                Beaucoupdeffort.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Beaucoupdeffort, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Beaucoupdeffort.started')
                # update status
                Beaucoupdeffort.status = STARTED
                win.callOnFlip(Beaucoupdeffort.buttonClock.reset)
                Beaucoupdeffort.setAutoDraw(True)
            
            # if Beaucoupdeffort is active this frame...
            if Beaucoupdeffort.status == STARTED:
                # update params
                pass
                # check whether Beaucoupdeffort has been pressed
                if Beaucoupdeffort.isClicked:
                    if not Beaucoupdeffort.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Beaucoupdeffort.timesOn.append(Beaucoupdeffort.buttonClock.getTime())
                        Beaucoupdeffort.timesOff.append(Beaucoupdeffort.buttonClock.getTime())
                    elif len(Beaucoupdeffort.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Beaucoupdeffort.timesOff[-1] = Beaucoupdeffort.buttonClock.getTime()
                    if not Beaucoupdeffort.wasClicked:
                        # end routine when Beaucoupdeffort is clicked
                        continueRoutine = False
                    if not Beaucoupdeffort.wasClicked:
                        # run callback code when Beaucoupdeffort is clicked
                        pass
            # take note of whether Beaucoupdeffort was clicked, so that next frame we know if clicks are new
            Beaucoupdeffort.wasClicked = Beaucoupdeffort.isClicked and Beaucoupdeffort.status == STARTED
            # *Quedubruit* updates
            
            # if Quedubruit is starting this frame...
            if Quedubruit.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Quedubruit.frameNStart = frameN  # exact frame index
                Quedubruit.tStart = t  # local t and not account for scr refresh
                Quedubruit.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Quedubruit, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Quedubruit.started')
                # update status
                Quedubruit.status = STARTED
                win.callOnFlip(Quedubruit.buttonClock.reset)
                Quedubruit.setAutoDraw(True)
            
            # if Quedubruit is active this frame...
            if Quedubruit.status == STARTED:
                # update params
                pass
                # check whether Quedubruit has been pressed
                if Quedubruit.isClicked:
                    if not Quedubruit.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Quedubruit.timesOn.append(Quedubruit.buttonClock.getTime())
                        Quedubruit.timesOff.append(Quedubruit.buttonClock.getTime())
                    elif len(Quedubruit.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Quedubruit.timesOff[-1] = Quedubruit.buttonClock.getTime()
                    if not Quedubruit.wasClicked:
                        # end routine when Quedubruit is clicked
                        continueRoutine = False
                    if not Quedubruit.wasClicked:
                        # run callback code when Quedubruit is clicked
                        pass
            # take note of whether Quedubruit was clicked, so that next frame we know if clicks are new
            Quedubruit.wasClicked = Quedubruit.isClicked and Quedubruit.status == STARTED
            
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
        trials_2loc.addData('Pasdeffort.numClicks', Pasdeffort.numClicks)
        if Pasdeffort.numClicks:
           trials_2loc.addData('Pasdeffort.timesOn', Pasdeffort.timesOn)
           trials_2loc.addData('Pasdeffort.timesOff', Pasdeffort.timesOff)
        else:
           trials_2loc.addData('Pasdeffort.timesOn', "")
           trials_2loc.addData('Pasdeffort.timesOff', "")
        trials_2loc.addData('__1.numClicks', __1.numClicks)
        if __1.numClicks:
           trials_2loc.addData('__1.timesOn', __1.timesOn)
           trials_2loc.addData('__1.timesOff', __1.timesOff)
        else:
           trials_2loc.addData('__1.timesOn', "")
           trials_2loc.addData('__1.timesOff', "")
        trials_2loc.addData('Trespeudeffort.numClicks', Trespeudeffort.numClicks)
        if Trespeudeffort.numClicks:
           trials_2loc.addData('Trespeudeffort.timesOn', Trespeudeffort.timesOn)
           trials_2loc.addData('Trespeudeffort.timesOff', Trespeudeffort.timesOff)
        else:
           trials_2loc.addData('Trespeudeffort.timesOn', "")
           trials_2loc.addData('Trespeudeffort.timesOff', "")
        trials_2loc.addData('__2.numClicks', __2.numClicks)
        if __2.numClicks:
           trials_2loc.addData('__2.timesOn', __2.timesOn)
           trials_2loc.addData('__2.timesOff', __2.timesOff)
        else:
           trials_2loc.addData('__2.timesOn', "")
           trials_2loc.addData('__2.timesOff', "")
        trials_2loc.addData('Peudeffort.numClicks', Peudeffort.numClicks)
        if Peudeffort.numClicks:
           trials_2loc.addData('Peudeffort.timesOn', Peudeffort.timesOn)
           trials_2loc.addData('Peudeffort.timesOff', Peudeffort.timesOff)
        else:
           trials_2loc.addData('Peudeffort.timesOn', "")
           trials_2loc.addData('Peudeffort.timesOff', "")
        trials_2loc.addData('__3.numClicks', __3.numClicks)
        if __3.numClicks:
           trials_2loc.addData('__3.timesOn', __3.timesOn)
           trials_2loc.addData('__3.timesOff', __3.timesOff)
        else:
           trials_2loc.addData('__3.timesOn', "")
           trials_2loc.addData('__3.timesOff', "")
        trials_2loc.addData('Effortmodere.numClicks', Effortmodere.numClicks)
        if Effortmodere.numClicks:
           trials_2loc.addData('Effortmodere.timesOn', Effortmodere.timesOn)
           trials_2loc.addData('Effortmodere.timesOff', Effortmodere.timesOff)
        else:
           trials_2loc.addData('Effortmodere.timesOn', "")
           trials_2loc.addData('Effortmodere.timesOff', "")
        trials_2loc.addData('__4.numClicks', __4.numClicks)
        if __4.numClicks:
           trials_2loc.addData('__4.timesOn', __4.timesOn)
           trials_2loc.addData('__4.timesOff', __4.timesOff)
        else:
           trials_2loc.addData('__4.timesOn', "")
           trials_2loc.addData('__4.timesOff', "")
        trials_2loc.addData('Effortconsiderable.numClicks', Effortconsiderable.numClicks)
        if Effortconsiderable.numClicks:
           trials_2loc.addData('Effortconsiderable.timesOn', Effortconsiderable.timesOn)
           trials_2loc.addData('Effortconsiderable.timesOff', Effortconsiderable.timesOff)
        else:
           trials_2loc.addData('Effortconsiderable.timesOn', "")
           trials_2loc.addData('Effortconsiderable.timesOff', "")
        trials_2loc.addData('__5.numClicks', __5.numClicks)
        if __5.numClicks:
           trials_2loc.addData('__5.timesOn', __5.timesOn)
           trials_2loc.addData('__5.timesOff', __5.timesOff)
        else:
           trials_2loc.addData('__5.timesOn', "")
           trials_2loc.addData('__5.timesOff', "")
        trials_2loc.addData('Beaucoupdeffort.numClicks', Beaucoupdeffort.numClicks)
        if Beaucoupdeffort.numClicks:
           trials_2loc.addData('Beaucoupdeffort.timesOn', Beaucoupdeffort.timesOn)
           trials_2loc.addData('Beaucoupdeffort.timesOff', Beaucoupdeffort.timesOff)
        else:
           trials_2loc.addData('Beaucoupdeffort.timesOn', "")
           trials_2loc.addData('Beaucoupdeffort.timesOff', "")
        trials_2loc.addData('Quedubruit.numClicks', Quedubruit.numClicks)
        if Quedubruit.numClicks:
           trials_2loc.addData('Quedubruit.timesOn', Quedubruit.timesOn)
           trials_2loc.addData('Quedubruit.timesOff', Quedubruit.timesOff)
        else:
           trials_2loc.addData('Quedubruit.timesOn', "")
           trials_2loc.addData('Quedubruit.timesOff', "")
        # the Routine "escu" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_2loc as finished
        if hasattr(thisTrials_2loc, 'status'):
            thisTrials_2loc.status = FINISHED
        # if awaiting a pause, pause now
        if trials_2loc.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_2loc.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'trials_2loc'
    trials_2loc.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Start_Bloc2" ---
    # create an object to store info about Routine Start_Bloc2
    Start_Bloc2 = data.Routine(
        name='Start_Bloc2',
        components=[StartBloc2, TextBloc2],
    )
    Start_Bloc2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset StartBloc2 to account for continued clicks & clear times on/off
    StartBloc2.reset()
    # store start times for Start_Bloc2
    Start_Bloc2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start_Bloc2.tStart = globalClock.getTime(format='float')
    Start_Bloc2.status = STARTED
    thisExp.addData('Start_Bloc2.started', Start_Bloc2.tStart)
    Start_Bloc2.maxDuration = None
    # keep track of which components have finished
    Start_Bloc2Components = Start_Bloc2.components
    for thisComponent in Start_Bloc2.components:
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
    
    # --- Run Routine "Start_Bloc2" ---
    thisExp.currentRoutine = Start_Bloc2
    Start_Bloc2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *StartBloc2* updates
        
        # if StartBloc2 is starting this frame...
        if StartBloc2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            StartBloc2.frameNStart = frameN  # exact frame index
            StartBloc2.tStart = t  # local t and not account for scr refresh
            StartBloc2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(StartBloc2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'StartBloc2.started')
            # update status
            StartBloc2.status = STARTED
            win.callOnFlip(StartBloc2.buttonClock.reset)
            StartBloc2.setAutoDraw(True)
        
        # if StartBloc2 is active this frame...
        if StartBloc2.status == STARTED:
            # update params
            pass
            # check whether StartBloc2 has been pressed
            if StartBloc2.isClicked:
                if not StartBloc2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    StartBloc2.timesOn.append(StartBloc2.buttonClock.getTime())
                    StartBloc2.timesOff.append(StartBloc2.buttonClock.getTime())
                elif len(StartBloc2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    StartBloc2.timesOff[-1] = StartBloc2.buttonClock.getTime()
                if not StartBloc2.wasClicked:
                    # end routine when StartBloc2 is clicked
                    continueRoutine = False
                if not StartBloc2.wasClicked:
                    # run callback code when StartBloc2 is clicked
                    pass
        # take note of whether StartBloc2 was clicked, so that next frame we know if clicks are new
        StartBloc2.wasClicked = StartBloc2.isClicked and StartBloc2.status == STARTED
        
        # *TextBloc2* updates
        
        # if TextBloc2 is starting this frame...
        if TextBloc2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            TextBloc2.frameNStart = frameN  # exact frame index
            TextBloc2.tStart = t  # local t and not account for scr refresh
            TextBloc2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(TextBloc2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'TextBloc2.started')
            # update status
            TextBloc2.status = STARTED
            TextBloc2.setAutoDraw(True)
        
        # if TextBloc2 is active this frame...
        if TextBloc2.status == STARTED:
            # update params
            pass
        
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
                currentRoutine=Start_Bloc2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Start_Bloc2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Start_Bloc2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Start_Bloc2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start_Bloc2" ---
    for thisComponent in Start_Bloc2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start_Bloc2
    Start_Bloc2.tStop = globalClock.getTime(format='float')
    Start_Bloc2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start_Bloc2.stopped', Start_Bloc2.tStop)
    thisExp.addData('StartBloc2.numClicks', StartBloc2.numClicks)
    if StartBloc2.numClicks:
       thisExp.addData('StartBloc2.timesOn', StartBloc2.timesOn)
       thisExp.addData('StartBloc2.timesOff', StartBloc2.timesOff)
    else:
       thisExp.addData('StartBloc2.timesOn', "")
       thisExp.addData('StartBloc2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "Start_Bloc2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_3loc = data.TrialHandler2(
        name='trials_3loc',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions/conditions_3loc.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials_3loc)  # add the loop to the experiment
    thisTrials_3loc = trials_3loc.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_3loc.rgb)
    if thisTrials_3loc != None:
        for paramName in thisTrials_3loc:
            globals()[paramName] = thisTrials_3loc[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_3loc in trials_3loc:
        trials_3loc.status = STARTED
        if hasattr(thisTrials_3loc, 'status'):
            thisTrials_3loc.status = STARTED
        currentLoop = trials_3loc
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_3loc.rgb)
        if thisTrials_3loc != None:
            for paramName in thisTrials_3loc:
                globals()[paramName] = thisTrials_3loc[paramName]
        
        # --- Prepare to start Routine "bloc_3loc" ---
        # create an object to store info about Routine bloc_3loc
        bloc_3loc = data.Routine(
            name='bloc_3loc',
            components=[text_instruction_3, R1_3, R2_3, R3_3, R4_3, R5_3, R6_3, R7_3, R8_3, B1_3, B2_3, B3_3, B4_3, B5_3, B6_3, B7_3, B8_3, V1_3, V2_3, V3_3, V4_3, V5_3, V6_3, V7_3, V8_3, J1_3, J2_3, J3_3, J4_3, J5_3, J6_3, J7_3, J8_3, Target_2, Marsker2_1, Masker2_2],
        )
        bloc_3loc.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # reset R1_3 to account for continued clicks & clear times on/off
        R1_3.reset()
        # reset R2_3 to account for continued clicks & clear times on/off
        R2_3.reset()
        # reset R3_3 to account for continued clicks & clear times on/off
        R3_3.reset()
        # reset R4_3 to account for continued clicks & clear times on/off
        R4_3.reset()
        # reset R5_3 to account for continued clicks & clear times on/off
        R5_3.reset()
        # reset R6_3 to account for continued clicks & clear times on/off
        R6_3.reset()
        # reset R7_3 to account for continued clicks & clear times on/off
        R7_3.reset()
        # reset R8_3 to account for continued clicks & clear times on/off
        R8_3.reset()
        # reset B1_3 to account for continued clicks & clear times on/off
        B1_3.reset()
        # reset B2_3 to account for continued clicks & clear times on/off
        B2_3.reset()
        # reset B3_3 to account for continued clicks & clear times on/off
        B3_3.reset()
        # reset B4_3 to account for continued clicks & clear times on/off
        B4_3.reset()
        # reset B5_3 to account for continued clicks & clear times on/off
        B5_3.reset()
        # reset B6_3 to account for continued clicks & clear times on/off
        B6_3.reset()
        # reset B7_3 to account for continued clicks & clear times on/off
        B7_3.reset()
        # reset B8_3 to account for continued clicks & clear times on/off
        B8_3.reset()
        # reset V1_3 to account for continued clicks & clear times on/off
        V1_3.reset()
        # reset V2_3 to account for continued clicks & clear times on/off
        V2_3.reset()
        # reset V3_3 to account for continued clicks & clear times on/off
        V3_3.reset()
        # reset V4_3 to account for continued clicks & clear times on/off
        V4_3.reset()
        # reset V5_3 to account for continued clicks & clear times on/off
        V5_3.reset()
        # reset V6_3 to account for continued clicks & clear times on/off
        V6_3.reset()
        # reset V7_3 to account for continued clicks & clear times on/off
        V7_3.reset()
        # reset V8_3 to account for continued clicks & clear times on/off
        V8_3.reset()
        # reset J1_3 to account for continued clicks & clear times on/off
        J1_3.reset()
        # reset J2_3 to account for continued clicks & clear times on/off
        J2_3.reset()
        # reset J3_3 to account for continued clicks & clear times on/off
        J3_3.reset()
        # reset J4_3 to account for continued clicks & clear times on/off
        J4_3.reset()
        # reset J5_3 to account for continued clicks & clear times on/off
        J5_3.reset()
        # reset J6_3 to account for continued clicks & clear times on/off
        J6_3.reset()
        # reset J7_3 to account for continued clicks & clear times on/off
        J7_3.reset()
        # reset J8_3 to account for continued clicks & clear times on/off
        J8_3.reset()
        Target_2.setSound(Target_3loc, hamming=True)
        Target_2.setVolume(1.0, log=False)
        Target_2.seek(0)
        Marsker2_1.setSound(Mask1_3loc, hamming=True)
        Marsker2_1.setVolume(1.0, log=False)
        Marsker2_1.seek(0)
        Masker2_2.setSound(Mask2_3loc, hamming=True)
        Masker2_2.setVolume(1.0, log=False)
        Masker2_2.seek(0)
        # store start times for bloc_3loc
        bloc_3loc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        bloc_3loc.tStart = globalClock.getTime(format='float')
        bloc_3loc.status = STARTED
        thisExp.addData('bloc_3loc.started', bloc_3loc.tStart)
        bloc_3loc.maxDuration = None
        # keep track of which components have finished
        bloc_3locComponents = bloc_3loc.components
        for thisComponent in bloc_3loc.components:
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
        
        # --- Run Routine "bloc_3loc" ---
        thisExp.currentRoutine = bloc_3loc
        bloc_3loc.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_3loc, 'status') and thisTrials_3loc.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_instruction_3* updates
            
            # if text_instruction_3 is starting this frame...
            if text_instruction_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_instruction_3.frameNStart = frameN  # exact frame index
                text_instruction_3.tStart = t  # local t and not account for scr refresh
                text_instruction_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_instruction_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_instruction_3.started')
                # update status
                text_instruction_3.status = STARTED
                text_instruction_3.setAutoDraw(True)
            
            # if text_instruction_3 is active this frame...
            if text_instruction_3.status == STARTED:
                # update params
                pass
            
            # if text_instruction_3 is stopping this frame...
            if text_instruction_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_instruction_3.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_instruction_3.tStop = t  # not accounting for scr refresh
                    text_instruction_3.tStopRefresh = tThisFlipGlobal  # on global time
                    text_instruction_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_instruction_3.stopped')
                    # update status
                    text_instruction_3.status = FINISHED
                    text_instruction_3.setAutoDraw(False)
            # *R1_3* updates
            
            # if R1_3 is starting this frame...
            if R1_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                R1_3.frameNStart = frameN  # exact frame index
                R1_3.tStart = t  # local t and not account for scr refresh
                R1_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R1_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R1_3.started')
                # update status
                R1_3.status = STARTED
                win.callOnFlip(R1_3.buttonClock.reset)
                R1_3.setAutoDraw(True)
            
            # if R1_3 is active this frame...
            if R1_3.status == STARTED:
                # update params
                pass
                # check whether R1_3 has been pressed
                if R1_3.isClicked:
                    if not R1_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        R1_3.timesOn.append(R1_3.buttonClock.getTime())
                        R1_3.timesOff.append(R1_3.buttonClock.getTime())
                    elif len(R1_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        R1_3.timesOff[-1] = R1_3.buttonClock.getTime()
                    if not R1_3.wasClicked:
                        # end routine when R1_3 is clicked
                        continueRoutine = False
                    if not R1_3.wasClicked:
                        # run callback code when R1_3 is clicked
                        pass
            # take note of whether R1_3 was clicked, so that next frame we know if clicks are new
            R1_3.wasClicked = R1_3.isClicked and R1_3.status == STARTED
            # *R2_3* updates
            
            # if R2_3 is starting this frame...
            if R2_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                R2_3.frameNStart = frameN  # exact frame index
                R2_3.tStart = t  # local t and not account for scr refresh
                R2_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R2_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R2_3.started')
                # update status
                R2_3.status = STARTED
                win.callOnFlip(R2_3.buttonClock.reset)
                R2_3.setAutoDraw(True)
            
            # if R2_3 is active this frame...
            if R2_3.status == STARTED:
                # update params
                pass
                # check whether R2_3 has been pressed
                if R2_3.isClicked:
                    if not R2_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        R2_3.timesOn.append(R2_3.buttonClock.getTime())
                        R2_3.timesOff.append(R2_3.buttonClock.getTime())
                    elif len(R2_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        R2_3.timesOff[-1] = R2_3.buttonClock.getTime()
                    if not R2_3.wasClicked:
                        # end routine when R2_3 is clicked
                        continueRoutine = False
                    if not R2_3.wasClicked:
                        # run callback code when R2_3 is clicked
                        pass
            # take note of whether R2_3 was clicked, so that next frame we know if clicks are new
            R2_3.wasClicked = R2_3.isClicked and R2_3.status == STARTED
            # *R3_3* updates
            
            # if R3_3 is starting this frame...
            if R3_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                R3_3.frameNStart = frameN  # exact frame index
                R3_3.tStart = t  # local t and not account for scr refresh
                R3_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R3_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R3_3.started')
                # update status
                R3_3.status = STARTED
                win.callOnFlip(R3_3.buttonClock.reset)
                R3_3.setAutoDraw(True)
            
            # if R3_3 is active this frame...
            if R3_3.status == STARTED:
                # update params
                pass
                # check whether R3_3 has been pressed
                if R3_3.isClicked:
                    if not R3_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        R3_3.timesOn.append(R3_3.buttonClock.getTime())
                        R3_3.timesOff.append(R3_3.buttonClock.getTime())
                    elif len(R3_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        R3_3.timesOff[-1] = R3_3.buttonClock.getTime()
                    if not R3_3.wasClicked:
                        # end routine when R3_3 is clicked
                        continueRoutine = False
                    if not R3_3.wasClicked:
                        # run callback code when R3_3 is clicked
                        pass
            # take note of whether R3_3 was clicked, so that next frame we know if clicks are new
            R3_3.wasClicked = R3_3.isClicked and R3_3.status == STARTED
            # *R4_3* updates
            
            # if R4_3 is starting this frame...
            if R4_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                R4_3.frameNStart = frameN  # exact frame index
                R4_3.tStart = t  # local t and not account for scr refresh
                R4_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R4_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R4_3.started')
                # update status
                R4_3.status = STARTED
                win.callOnFlip(R4_3.buttonClock.reset)
                R4_3.setAutoDraw(True)
            
            # if R4_3 is active this frame...
            if R4_3.status == STARTED:
                # update params
                pass
                # check whether R4_3 has been pressed
                if R4_3.isClicked:
                    if not R4_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        R4_3.timesOn.append(R4_3.buttonClock.getTime())
                        R4_3.timesOff.append(R4_3.buttonClock.getTime())
                    elif len(R4_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        R4_3.timesOff[-1] = R4_3.buttonClock.getTime()
                    if not R4_3.wasClicked:
                        # end routine when R4_3 is clicked
                        continueRoutine = False
                    if not R4_3.wasClicked:
                        # run callback code when R4_3 is clicked
                        pass
            # take note of whether R4_3 was clicked, so that next frame we know if clicks are new
            R4_3.wasClicked = R4_3.isClicked and R4_3.status == STARTED
            # *R5_3* updates
            
            # if R5_3 is starting this frame...
            if R5_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                R5_3.frameNStart = frameN  # exact frame index
                R5_3.tStart = t  # local t and not account for scr refresh
                R5_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R5_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R5_3.started')
                # update status
                R5_3.status = STARTED
                win.callOnFlip(R5_3.buttonClock.reset)
                R5_3.setAutoDraw(True)
            
            # if R5_3 is active this frame...
            if R5_3.status == STARTED:
                # update params
                pass
                # check whether R5_3 has been pressed
                if R5_3.isClicked:
                    if not R5_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        R5_3.timesOn.append(R5_3.buttonClock.getTime())
                        R5_3.timesOff.append(R5_3.buttonClock.getTime())
                    elif len(R5_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        R5_3.timesOff[-1] = R5_3.buttonClock.getTime()
                    if not R5_3.wasClicked:
                        # end routine when R5_3 is clicked
                        continueRoutine = False
                    if not R5_3.wasClicked:
                        # run callback code when R5_3 is clicked
                        pass
            # take note of whether R5_3 was clicked, so that next frame we know if clicks are new
            R5_3.wasClicked = R5_3.isClicked and R5_3.status == STARTED
            # *R6_3* updates
            
            # if R6_3 is starting this frame...
            if R6_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                R6_3.frameNStart = frameN  # exact frame index
                R6_3.tStart = t  # local t and not account for scr refresh
                R6_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R6_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R6_3.started')
                # update status
                R6_3.status = STARTED
                win.callOnFlip(R6_3.buttonClock.reset)
                R6_3.setAutoDraw(True)
            
            # if R6_3 is active this frame...
            if R6_3.status == STARTED:
                # update params
                pass
                # check whether R6_3 has been pressed
                if R6_3.isClicked:
                    if not R6_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        R6_3.timesOn.append(R6_3.buttonClock.getTime())
                        R6_3.timesOff.append(R6_3.buttonClock.getTime())
                    elif len(R6_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        R6_3.timesOff[-1] = R6_3.buttonClock.getTime()
                    if not R6_3.wasClicked:
                        # end routine when R6_3 is clicked
                        continueRoutine = False
                    if not R6_3.wasClicked:
                        # run callback code when R6_3 is clicked
                        pass
            # take note of whether R6_3 was clicked, so that next frame we know if clicks are new
            R6_3.wasClicked = R6_3.isClicked and R6_3.status == STARTED
            # *R7_3* updates
            
            # if R7_3 is starting this frame...
            if R7_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                R7_3.frameNStart = frameN  # exact frame index
                R7_3.tStart = t  # local t and not account for scr refresh
                R7_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R7_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R7_3.started')
                # update status
                R7_3.status = STARTED
                win.callOnFlip(R7_3.buttonClock.reset)
                R7_3.setAutoDraw(True)
            
            # if R7_3 is active this frame...
            if R7_3.status == STARTED:
                # update params
                pass
                # check whether R7_3 has been pressed
                if R7_3.isClicked:
                    if not R7_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        R7_3.timesOn.append(R7_3.buttonClock.getTime())
                        R7_3.timesOff.append(R7_3.buttonClock.getTime())
                    elif len(R7_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        R7_3.timesOff[-1] = R7_3.buttonClock.getTime()
                    if not R7_3.wasClicked:
                        # end routine when R7_3 is clicked
                        continueRoutine = False
                    if not R7_3.wasClicked:
                        # run callback code when R7_3 is clicked
                        pass
            # take note of whether R7_3 was clicked, so that next frame we know if clicks are new
            R7_3.wasClicked = R7_3.isClicked and R7_3.status == STARTED
            # *R8_3* updates
            
            # if R8_3 is starting this frame...
            if R8_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                R8_3.frameNStart = frameN  # exact frame index
                R8_3.tStart = t  # local t and not account for scr refresh
                R8_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(R8_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'R8_3.started')
                # update status
                R8_3.status = STARTED
                win.callOnFlip(R8_3.buttonClock.reset)
                R8_3.setAutoDraw(True)
            
            # if R8_3 is active this frame...
            if R8_3.status == STARTED:
                # update params
                pass
                # check whether R8_3 has been pressed
                if R8_3.isClicked:
                    if not R8_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        R8_3.timesOn.append(R8_3.buttonClock.getTime())
                        R8_3.timesOff.append(R8_3.buttonClock.getTime())
                    elif len(R8_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        R8_3.timesOff[-1] = R8_3.buttonClock.getTime()
                    if not R8_3.wasClicked:
                        # end routine when R8_3 is clicked
                        continueRoutine = False
                    if not R8_3.wasClicked:
                        # run callback code when R8_3 is clicked
                        pass
            # take note of whether R8_3 was clicked, so that next frame we know if clicks are new
            R8_3.wasClicked = R8_3.isClicked and R8_3.status == STARTED
            # *B1_3* updates
            
            # if B1_3 is starting this frame...
            if B1_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                B1_3.frameNStart = frameN  # exact frame index
                B1_3.tStart = t  # local t and not account for scr refresh
                B1_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B1_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B1_3.started')
                # update status
                B1_3.status = STARTED
                win.callOnFlip(B1_3.buttonClock.reset)
                B1_3.setAutoDraw(True)
            
            # if B1_3 is active this frame...
            if B1_3.status == STARTED:
                # update params
                pass
                # check whether B1_3 has been pressed
                if B1_3.isClicked:
                    if not B1_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        B1_3.timesOn.append(B1_3.buttonClock.getTime())
                        B1_3.timesOff.append(B1_3.buttonClock.getTime())
                    elif len(B1_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        B1_3.timesOff[-1] = B1_3.buttonClock.getTime()
                    if not B1_3.wasClicked:
                        # end routine when B1_3 is clicked
                        continueRoutine = False
                    if not B1_3.wasClicked:
                        # run callback code when B1_3 is clicked
                        pass
            # take note of whether B1_3 was clicked, so that next frame we know if clicks are new
            B1_3.wasClicked = B1_3.isClicked and B1_3.status == STARTED
            # *B2_3* updates
            
            # if B2_3 is starting this frame...
            if B2_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                B2_3.frameNStart = frameN  # exact frame index
                B2_3.tStart = t  # local t and not account for scr refresh
                B2_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B2_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B2_3.started')
                # update status
                B2_3.status = STARTED
                win.callOnFlip(B2_3.buttonClock.reset)
                B2_3.setAutoDraw(True)
            
            # if B2_3 is active this frame...
            if B2_3.status == STARTED:
                # update params
                pass
                # check whether B2_3 has been pressed
                if B2_3.isClicked:
                    if not B2_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        B2_3.timesOn.append(B2_3.buttonClock.getTime())
                        B2_3.timesOff.append(B2_3.buttonClock.getTime())
                    elif len(B2_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        B2_3.timesOff[-1] = B2_3.buttonClock.getTime()
                    if not B2_3.wasClicked:
                        # end routine when B2_3 is clicked
                        continueRoutine = False
                    if not B2_3.wasClicked:
                        # run callback code when B2_3 is clicked
                        pass
            # take note of whether B2_3 was clicked, so that next frame we know if clicks are new
            B2_3.wasClicked = B2_3.isClicked and B2_3.status == STARTED
            # *B3_3* updates
            
            # if B3_3 is starting this frame...
            if B3_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                B3_3.frameNStart = frameN  # exact frame index
                B3_3.tStart = t  # local t and not account for scr refresh
                B3_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B3_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B3_3.started')
                # update status
                B3_3.status = STARTED
                win.callOnFlip(B3_3.buttonClock.reset)
                B3_3.setAutoDraw(True)
            
            # if B3_3 is active this frame...
            if B3_3.status == STARTED:
                # update params
                pass
                # check whether B3_3 has been pressed
                if B3_3.isClicked:
                    if not B3_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        B3_3.timesOn.append(B3_3.buttonClock.getTime())
                        B3_3.timesOff.append(B3_3.buttonClock.getTime())
                    elif len(B3_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        B3_3.timesOff[-1] = B3_3.buttonClock.getTime()
                    if not B3_3.wasClicked:
                        # end routine when B3_3 is clicked
                        continueRoutine = False
                    if not B3_3.wasClicked:
                        # run callback code when B3_3 is clicked
                        pass
            # take note of whether B3_3 was clicked, so that next frame we know if clicks are new
            B3_3.wasClicked = B3_3.isClicked and B3_3.status == STARTED
            # *B4_3* updates
            
            # if B4_3 is starting this frame...
            if B4_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                B4_3.frameNStart = frameN  # exact frame index
                B4_3.tStart = t  # local t and not account for scr refresh
                B4_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B4_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B4_3.started')
                # update status
                B4_3.status = STARTED
                win.callOnFlip(B4_3.buttonClock.reset)
                B4_3.setAutoDraw(True)
            
            # if B4_3 is active this frame...
            if B4_3.status == STARTED:
                # update params
                pass
                # check whether B4_3 has been pressed
                if B4_3.isClicked:
                    if not B4_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        B4_3.timesOn.append(B4_3.buttonClock.getTime())
                        B4_3.timesOff.append(B4_3.buttonClock.getTime())
                    elif len(B4_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        B4_3.timesOff[-1] = B4_3.buttonClock.getTime()
                    if not B4_3.wasClicked:
                        # end routine when B4_3 is clicked
                        continueRoutine = False
                    if not B4_3.wasClicked:
                        # run callback code when B4_3 is clicked
                        pass
            # take note of whether B4_3 was clicked, so that next frame we know if clicks are new
            B4_3.wasClicked = B4_3.isClicked and B4_3.status == STARTED
            # *B5_3* updates
            
            # if B5_3 is starting this frame...
            if B5_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                B5_3.frameNStart = frameN  # exact frame index
                B5_3.tStart = t  # local t and not account for scr refresh
                B5_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B5_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B5_3.started')
                # update status
                B5_3.status = STARTED
                win.callOnFlip(B5_3.buttonClock.reset)
                B5_3.setAutoDraw(True)
            
            # if B5_3 is active this frame...
            if B5_3.status == STARTED:
                # update params
                pass
                # check whether B5_3 has been pressed
                if B5_3.isClicked:
                    if not B5_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        B5_3.timesOn.append(B5_3.buttonClock.getTime())
                        B5_3.timesOff.append(B5_3.buttonClock.getTime())
                    elif len(B5_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        B5_3.timesOff[-1] = B5_3.buttonClock.getTime()
                    if not B5_3.wasClicked:
                        # end routine when B5_3 is clicked
                        continueRoutine = False
                    if not B5_3.wasClicked:
                        # run callback code when B5_3 is clicked
                        pass
            # take note of whether B5_3 was clicked, so that next frame we know if clicks are new
            B5_3.wasClicked = B5_3.isClicked and B5_3.status == STARTED
            # *B6_3* updates
            
            # if B6_3 is starting this frame...
            if B6_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                B6_3.frameNStart = frameN  # exact frame index
                B6_3.tStart = t  # local t and not account for scr refresh
                B6_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B6_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B6_3.started')
                # update status
                B6_3.status = STARTED
                win.callOnFlip(B6_3.buttonClock.reset)
                B6_3.setAutoDraw(True)
            
            # if B6_3 is active this frame...
            if B6_3.status == STARTED:
                # update params
                pass
                # check whether B6_3 has been pressed
                if B6_3.isClicked:
                    if not B6_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        B6_3.timesOn.append(B6_3.buttonClock.getTime())
                        B6_3.timesOff.append(B6_3.buttonClock.getTime())
                    elif len(B6_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        B6_3.timesOff[-1] = B6_3.buttonClock.getTime()
                    if not B6_3.wasClicked:
                        # end routine when B6_3 is clicked
                        continueRoutine = False
                    if not B6_3.wasClicked:
                        # run callback code when B6_3 is clicked
                        pass
            # take note of whether B6_3 was clicked, so that next frame we know if clicks are new
            B6_3.wasClicked = B6_3.isClicked and B6_3.status == STARTED
            # *B7_3* updates
            
            # if B7_3 is starting this frame...
            if B7_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                B7_3.frameNStart = frameN  # exact frame index
                B7_3.tStart = t  # local t and not account for scr refresh
                B7_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B7_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B7_3.started')
                # update status
                B7_3.status = STARTED
                win.callOnFlip(B7_3.buttonClock.reset)
                B7_3.setAutoDraw(True)
            
            # if B7_3 is active this frame...
            if B7_3.status == STARTED:
                # update params
                pass
                # check whether B7_3 has been pressed
                if B7_3.isClicked:
                    if not B7_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        B7_3.timesOn.append(B7_3.buttonClock.getTime())
                        B7_3.timesOff.append(B7_3.buttonClock.getTime())
                    elif len(B7_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        B7_3.timesOff[-1] = B7_3.buttonClock.getTime()
                    if not B7_3.wasClicked:
                        # end routine when B7_3 is clicked
                        continueRoutine = False
                    if not B7_3.wasClicked:
                        # run callback code when B7_3 is clicked
                        pass
            # take note of whether B7_3 was clicked, so that next frame we know if clicks are new
            B7_3.wasClicked = B7_3.isClicked and B7_3.status == STARTED
            # *B8_3* updates
            
            # if B8_3 is starting this frame...
            if B8_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                B8_3.frameNStart = frameN  # exact frame index
                B8_3.tStart = t  # local t and not account for scr refresh
                B8_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(B8_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'B8_3.started')
                # update status
                B8_3.status = STARTED
                win.callOnFlip(B8_3.buttonClock.reset)
                B8_3.setAutoDraw(True)
            
            # if B8_3 is active this frame...
            if B8_3.status == STARTED:
                # update params
                pass
                # check whether B8_3 has been pressed
                if B8_3.isClicked:
                    if not B8_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        B8_3.timesOn.append(B8_3.buttonClock.getTime())
                        B8_3.timesOff.append(B8_3.buttonClock.getTime())
                    elif len(B8_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        B8_3.timesOff[-1] = B8_3.buttonClock.getTime()
                    if not B8_3.wasClicked:
                        # end routine when B8_3 is clicked
                        continueRoutine = False
                    if not B8_3.wasClicked:
                        # run callback code when B8_3 is clicked
                        pass
            # take note of whether B8_3 was clicked, so that next frame we know if clicks are new
            B8_3.wasClicked = B8_3.isClicked and B8_3.status == STARTED
            # *V1_3* updates
            
            # if V1_3 is starting this frame...
            if V1_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                V1_3.frameNStart = frameN  # exact frame index
                V1_3.tStart = t  # local t and not account for scr refresh
                V1_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(V1_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'V1_3.started')
                # update status
                V1_3.status = STARTED
                win.callOnFlip(V1_3.buttonClock.reset)
                V1_3.setAutoDraw(True)
            
            # if V1_3 is active this frame...
            if V1_3.status == STARTED:
                # update params
                pass
                # check whether V1_3 has been pressed
                if V1_3.isClicked:
                    if not V1_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        V1_3.timesOn.append(V1_3.buttonClock.getTime())
                        V1_3.timesOff.append(V1_3.buttonClock.getTime())
                    elif len(V1_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        V1_3.timesOff[-1] = V1_3.buttonClock.getTime()
                    if not V1_3.wasClicked:
                        # end routine when V1_3 is clicked
                        continueRoutine = False
                    if not V1_3.wasClicked:
                        # run callback code when V1_3 is clicked
                        pass
            # take note of whether V1_3 was clicked, so that next frame we know if clicks are new
            V1_3.wasClicked = V1_3.isClicked and V1_3.status == STARTED
            # *V2_3* updates
            
            # if V2_3 is starting this frame...
            if V2_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                V2_3.frameNStart = frameN  # exact frame index
                V2_3.tStart = t  # local t and not account for scr refresh
                V2_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(V2_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'V2_3.started')
                # update status
                V2_3.status = STARTED
                win.callOnFlip(V2_3.buttonClock.reset)
                V2_3.setAutoDraw(True)
            
            # if V2_3 is active this frame...
            if V2_3.status == STARTED:
                # update params
                pass
                # check whether V2_3 has been pressed
                if V2_3.isClicked:
                    if not V2_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        V2_3.timesOn.append(V2_3.buttonClock.getTime())
                        V2_3.timesOff.append(V2_3.buttonClock.getTime())
                    elif len(V2_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        V2_3.timesOff[-1] = V2_3.buttonClock.getTime()
                    if not V2_3.wasClicked:
                        # end routine when V2_3 is clicked
                        continueRoutine = False
                    if not V2_3.wasClicked:
                        # run callback code when V2_3 is clicked
                        pass
            # take note of whether V2_3 was clicked, so that next frame we know if clicks are new
            V2_3.wasClicked = V2_3.isClicked and V2_3.status == STARTED
            # *V3_3* updates
            
            # if V3_3 is starting this frame...
            if V3_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                V3_3.frameNStart = frameN  # exact frame index
                V3_3.tStart = t  # local t and not account for scr refresh
                V3_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(V3_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'V3_3.started')
                # update status
                V3_3.status = STARTED
                win.callOnFlip(V3_3.buttonClock.reset)
                V3_3.setAutoDraw(True)
            
            # if V3_3 is active this frame...
            if V3_3.status == STARTED:
                # update params
                pass
                # check whether V3_3 has been pressed
                if V3_3.isClicked:
                    if not V3_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        V3_3.timesOn.append(V3_3.buttonClock.getTime())
                        V3_3.timesOff.append(V3_3.buttonClock.getTime())
                    elif len(V3_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        V3_3.timesOff[-1] = V3_3.buttonClock.getTime()
                    if not V3_3.wasClicked:
                        # end routine when V3_3 is clicked
                        continueRoutine = False
                    if not V3_3.wasClicked:
                        # run callback code when V3_3 is clicked
                        pass
            # take note of whether V3_3 was clicked, so that next frame we know if clicks are new
            V3_3.wasClicked = V3_3.isClicked and V3_3.status == STARTED
            # *V4_3* updates
            
            # if V4_3 is starting this frame...
            if V4_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                V4_3.frameNStart = frameN  # exact frame index
                V4_3.tStart = t  # local t and not account for scr refresh
                V4_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(V4_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'V4_3.started')
                # update status
                V4_3.status = STARTED
                win.callOnFlip(V4_3.buttonClock.reset)
                V4_3.setAutoDraw(True)
            
            # if V4_3 is active this frame...
            if V4_3.status == STARTED:
                # update params
                pass
                # check whether V4_3 has been pressed
                if V4_3.isClicked:
                    if not V4_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        V4_3.timesOn.append(V4_3.buttonClock.getTime())
                        V4_3.timesOff.append(V4_3.buttonClock.getTime())
                    elif len(V4_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        V4_3.timesOff[-1] = V4_3.buttonClock.getTime()
                    if not V4_3.wasClicked:
                        # end routine when V4_3 is clicked
                        continueRoutine = False
                    if not V4_3.wasClicked:
                        # run callback code when V4_3 is clicked
                        pass
            # take note of whether V4_3 was clicked, so that next frame we know if clicks are new
            V4_3.wasClicked = V4_3.isClicked and V4_3.status == STARTED
            # *V5_3* updates
            
            # if V5_3 is starting this frame...
            if V5_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                V5_3.frameNStart = frameN  # exact frame index
                V5_3.tStart = t  # local t and not account for scr refresh
                V5_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(V5_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'V5_3.started')
                # update status
                V5_3.status = STARTED
                win.callOnFlip(V5_3.buttonClock.reset)
                V5_3.setAutoDraw(True)
            
            # if V5_3 is active this frame...
            if V5_3.status == STARTED:
                # update params
                pass
                # check whether V5_3 has been pressed
                if V5_3.isClicked:
                    if not V5_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        V5_3.timesOn.append(V5_3.buttonClock.getTime())
                        V5_3.timesOff.append(V5_3.buttonClock.getTime())
                    elif len(V5_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        V5_3.timesOff[-1] = V5_3.buttonClock.getTime()
                    if not V5_3.wasClicked:
                        # end routine when V5_3 is clicked
                        continueRoutine = False
                    if not V5_3.wasClicked:
                        # run callback code when V5_3 is clicked
                        pass
            # take note of whether V5_3 was clicked, so that next frame we know if clicks are new
            V5_3.wasClicked = V5_3.isClicked and V5_3.status == STARTED
            # *V6_3* updates
            
            # if V6_3 is starting this frame...
            if V6_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                V6_3.frameNStart = frameN  # exact frame index
                V6_3.tStart = t  # local t and not account for scr refresh
                V6_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(V6_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'V6_3.started')
                # update status
                V6_3.status = STARTED
                win.callOnFlip(V6_3.buttonClock.reset)
                V6_3.setAutoDraw(True)
            
            # if V6_3 is active this frame...
            if V6_3.status == STARTED:
                # update params
                pass
                # check whether V6_3 has been pressed
                if V6_3.isClicked:
                    if not V6_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        V6_3.timesOn.append(V6_3.buttonClock.getTime())
                        V6_3.timesOff.append(V6_3.buttonClock.getTime())
                    elif len(V6_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        V6_3.timesOff[-1] = V6_3.buttonClock.getTime()
                    if not V6_3.wasClicked:
                        # end routine when V6_3 is clicked
                        continueRoutine = False
                    if not V6_3.wasClicked:
                        # run callback code when V6_3 is clicked
                        pass
            # take note of whether V6_3 was clicked, so that next frame we know if clicks are new
            V6_3.wasClicked = V6_3.isClicked and V6_3.status == STARTED
            # *V7_3* updates
            
            # if V7_3 is starting this frame...
            if V7_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                V7_3.frameNStart = frameN  # exact frame index
                V7_3.tStart = t  # local t and not account for scr refresh
                V7_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(V7_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'V7_3.started')
                # update status
                V7_3.status = STARTED
                win.callOnFlip(V7_3.buttonClock.reset)
                V7_3.setAutoDraw(True)
            
            # if V7_3 is active this frame...
            if V7_3.status == STARTED:
                # update params
                pass
                # check whether V7_3 has been pressed
                if V7_3.isClicked:
                    if not V7_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        V7_3.timesOn.append(V7_3.buttonClock.getTime())
                        V7_3.timesOff.append(V7_3.buttonClock.getTime())
                    elif len(V7_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        V7_3.timesOff[-1] = V7_3.buttonClock.getTime()
                    if not V7_3.wasClicked:
                        # end routine when V7_3 is clicked
                        continueRoutine = False
                    if not V7_3.wasClicked:
                        # run callback code when V7_3 is clicked
                        pass
            # take note of whether V7_3 was clicked, so that next frame we know if clicks are new
            V7_3.wasClicked = V7_3.isClicked and V7_3.status == STARTED
            # *V8_3* updates
            
            # if V8_3 is starting this frame...
            if V8_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                V8_3.frameNStart = frameN  # exact frame index
                V8_3.tStart = t  # local t and not account for scr refresh
                V8_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(V8_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'V8_3.started')
                # update status
                V8_3.status = STARTED
                win.callOnFlip(V8_3.buttonClock.reset)
                V8_3.setAutoDraw(True)
            
            # if V8_3 is active this frame...
            if V8_3.status == STARTED:
                # update params
                pass
                # check whether V8_3 has been pressed
                if V8_3.isClicked:
                    if not V8_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        V8_3.timesOn.append(V8_3.buttonClock.getTime())
                        V8_3.timesOff.append(V8_3.buttonClock.getTime())
                    elif len(V8_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        V8_3.timesOff[-1] = V8_3.buttonClock.getTime()
                    if not V8_3.wasClicked:
                        # end routine when V8_3 is clicked
                        continueRoutine = False
                    if not V8_3.wasClicked:
                        # run callback code when V8_3 is clicked
                        pass
            # take note of whether V8_3 was clicked, so that next frame we know if clicks are new
            V8_3.wasClicked = V8_3.isClicked and V8_3.status == STARTED
            # *J1_3* updates
            
            # if J1_3 is starting this frame...
            if J1_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                J1_3.frameNStart = frameN  # exact frame index
                J1_3.tStart = t  # local t and not account for scr refresh
                J1_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J1_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'J1_3.started')
                # update status
                J1_3.status = STARTED
                win.callOnFlip(J1_3.buttonClock.reset)
                J1_3.setAutoDraw(True)
            
            # if J1_3 is active this frame...
            if J1_3.status == STARTED:
                # update params
                pass
                # check whether J1_3 has been pressed
                if J1_3.isClicked:
                    if not J1_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        J1_3.timesOn.append(J1_3.buttonClock.getTime())
                        J1_3.timesOff.append(J1_3.buttonClock.getTime())
                    elif len(J1_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        J1_3.timesOff[-1] = J1_3.buttonClock.getTime()
                    if not J1_3.wasClicked:
                        # end routine when J1_3 is clicked
                        continueRoutine = False
                    if not J1_3.wasClicked:
                        # run callback code when J1_3 is clicked
                        pass
            # take note of whether J1_3 was clicked, so that next frame we know if clicks are new
            J1_3.wasClicked = J1_3.isClicked and J1_3.status == STARTED
            # *J2_3* updates
            
            # if J2_3 is starting this frame...
            if J2_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                J2_3.frameNStart = frameN  # exact frame index
                J2_3.tStart = t  # local t and not account for scr refresh
                J2_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J2_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'J2_3.started')
                # update status
                J2_3.status = STARTED
                win.callOnFlip(J2_3.buttonClock.reset)
                J2_3.setAutoDraw(True)
            
            # if J2_3 is active this frame...
            if J2_3.status == STARTED:
                # update params
                pass
                # check whether J2_3 has been pressed
                if J2_3.isClicked:
                    if not J2_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        J2_3.timesOn.append(J2_3.buttonClock.getTime())
                        J2_3.timesOff.append(J2_3.buttonClock.getTime())
                    elif len(J2_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        J2_3.timesOff[-1] = J2_3.buttonClock.getTime()
                    if not J2_3.wasClicked:
                        # end routine when J2_3 is clicked
                        continueRoutine = False
                    if not J2_3.wasClicked:
                        # run callback code when J2_3 is clicked
                        pass
            # take note of whether J2_3 was clicked, so that next frame we know if clicks are new
            J2_3.wasClicked = J2_3.isClicked and J2_3.status == STARTED
            # *J3_3* updates
            
            # if J3_3 is starting this frame...
            if J3_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                J3_3.frameNStart = frameN  # exact frame index
                J3_3.tStart = t  # local t and not account for scr refresh
                J3_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J3_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'J3_3.started')
                # update status
                J3_3.status = STARTED
                win.callOnFlip(J3_3.buttonClock.reset)
                J3_3.setAutoDraw(True)
            
            # if J3_3 is active this frame...
            if J3_3.status == STARTED:
                # update params
                pass
                # check whether J3_3 has been pressed
                if J3_3.isClicked:
                    if not J3_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        J3_3.timesOn.append(J3_3.buttonClock.getTime())
                        J3_3.timesOff.append(J3_3.buttonClock.getTime())
                    elif len(J3_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        J3_3.timesOff[-1] = J3_3.buttonClock.getTime()
                    if not J3_3.wasClicked:
                        # end routine when J3_3 is clicked
                        continueRoutine = False
                    if not J3_3.wasClicked:
                        # run callback code when J3_3 is clicked
                        pass
            # take note of whether J3_3 was clicked, so that next frame we know if clicks are new
            J3_3.wasClicked = J3_3.isClicked and J3_3.status == STARTED
            # *J4_3* updates
            
            # if J4_3 is starting this frame...
            if J4_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                J4_3.frameNStart = frameN  # exact frame index
                J4_3.tStart = t  # local t and not account for scr refresh
                J4_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J4_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'J4_3.started')
                # update status
                J4_3.status = STARTED
                win.callOnFlip(J4_3.buttonClock.reset)
                J4_3.setAutoDraw(True)
            
            # if J4_3 is active this frame...
            if J4_3.status == STARTED:
                # update params
                pass
                # check whether J4_3 has been pressed
                if J4_3.isClicked:
                    if not J4_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        J4_3.timesOn.append(J4_3.buttonClock.getTime())
                        J4_3.timesOff.append(J4_3.buttonClock.getTime())
                    elif len(J4_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        J4_3.timesOff[-1] = J4_3.buttonClock.getTime()
                    if not J4_3.wasClicked:
                        # end routine when J4_3 is clicked
                        continueRoutine = False
                    if not J4_3.wasClicked:
                        # run callback code when J4_3 is clicked
                        pass
            # take note of whether J4_3 was clicked, so that next frame we know if clicks are new
            J4_3.wasClicked = J4_3.isClicked and J4_3.status == STARTED
            # *J5_3* updates
            
            # if J5_3 is starting this frame...
            if J5_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                J5_3.frameNStart = frameN  # exact frame index
                J5_3.tStart = t  # local t and not account for scr refresh
                J5_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J5_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'J5_3.started')
                # update status
                J5_3.status = STARTED
                win.callOnFlip(J5_3.buttonClock.reset)
                J5_3.setAutoDraw(True)
            
            # if J5_3 is active this frame...
            if J5_3.status == STARTED:
                # update params
                pass
                # check whether J5_3 has been pressed
                if J5_3.isClicked:
                    if not J5_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        J5_3.timesOn.append(J5_3.buttonClock.getTime())
                        J5_3.timesOff.append(J5_3.buttonClock.getTime())
                    elif len(J5_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        J5_3.timesOff[-1] = J5_3.buttonClock.getTime()
                    if not J5_3.wasClicked:
                        # end routine when J5_3 is clicked
                        continueRoutine = False
                    if not J5_3.wasClicked:
                        # run callback code when J5_3 is clicked
                        pass
            # take note of whether J5_3 was clicked, so that next frame we know if clicks are new
            J5_3.wasClicked = J5_3.isClicked and J5_3.status == STARTED
            # *J6_3* updates
            
            # if J6_3 is starting this frame...
            if J6_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                J6_3.frameNStart = frameN  # exact frame index
                J6_3.tStart = t  # local t and not account for scr refresh
                J6_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J6_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'J6_3.started')
                # update status
                J6_3.status = STARTED
                win.callOnFlip(J6_3.buttonClock.reset)
                J6_3.setAutoDraw(True)
            
            # if J6_3 is active this frame...
            if J6_3.status == STARTED:
                # update params
                pass
                # check whether J6_3 has been pressed
                if J6_3.isClicked:
                    if not J6_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        J6_3.timesOn.append(J6_3.buttonClock.getTime())
                        J6_3.timesOff.append(J6_3.buttonClock.getTime())
                    elif len(J6_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        J6_3.timesOff[-1] = J6_3.buttonClock.getTime()
                    if not J6_3.wasClicked:
                        # end routine when J6_3 is clicked
                        continueRoutine = False
                    if not J6_3.wasClicked:
                        # run callback code when J6_3 is clicked
                        pass
            # take note of whether J6_3 was clicked, so that next frame we know if clicks are new
            J6_3.wasClicked = J6_3.isClicked and J6_3.status == STARTED
            # *J7_3* updates
            
            # if J7_3 is starting this frame...
            if J7_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                J7_3.frameNStart = frameN  # exact frame index
                J7_3.tStart = t  # local t and not account for scr refresh
                J7_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J7_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'J7_3.started')
                # update status
                J7_3.status = STARTED
                win.callOnFlip(J7_3.buttonClock.reset)
                J7_3.setAutoDraw(True)
            
            # if J7_3 is active this frame...
            if J7_3.status == STARTED:
                # update params
                pass
                # check whether J7_3 has been pressed
                if J7_3.isClicked:
                    if not J7_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        J7_3.timesOn.append(J7_3.buttonClock.getTime())
                        J7_3.timesOff.append(J7_3.buttonClock.getTime())
                    elif len(J7_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        J7_3.timesOff[-1] = J7_3.buttonClock.getTime()
                    if not J7_3.wasClicked:
                        # end routine when J7_3 is clicked
                        continueRoutine = False
                    if not J7_3.wasClicked:
                        # run callback code when J7_3 is clicked
                        pass
            # take note of whether J7_3 was clicked, so that next frame we know if clicks are new
            J7_3.wasClicked = J7_3.isClicked and J7_3.status == STARTED
            # *J8_3* updates
            
            # if J8_3 is starting this frame...
            if J8_3.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                J8_3.frameNStart = frameN  # exact frame index
                J8_3.tStart = t  # local t and not account for scr refresh
                J8_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J8_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'J8_3.started')
                # update status
                J8_3.status = STARTED
                win.callOnFlip(J8_3.buttonClock.reset)
                J8_3.setAutoDraw(True)
            
            # if J8_3 is active this frame...
            if J8_3.status == STARTED:
                # update params
                pass
                # check whether J8_3 has been pressed
                if J8_3.isClicked:
                    if not J8_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        J8_3.timesOn.append(J8_3.buttonClock.getTime())
                        J8_3.timesOff.append(J8_3.buttonClock.getTime())
                    elif len(J8_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        J8_3.timesOff[-1] = J8_3.buttonClock.getTime()
                    if not J8_3.wasClicked:
                        # end routine when J8_3 is clicked
                        continueRoutine = False
                    if not J8_3.wasClicked:
                        # run callback code when J8_3 is clicked
                        pass
            # take note of whether J8_3 was clicked, so that next frame we know if clicks are new
            J8_3.wasClicked = J8_3.isClicked and J8_3.status == STARTED
            
            # *Target_2* updates
            
            # if Target_2 is starting this frame...
            if Target_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                Target_2.frameNStart = frameN  # exact frame index
                Target_2.tStart = t  # local t and not account for scr refresh
                Target_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('Target_2.started', tThisFlipGlobal)
                # update status
                Target_2.status = STARTED
                Target_2.play(when=win)  # sync with win flip
            
            # if Target_2 is stopping this frame...
            if Target_2.status == STARTED:
                if bool(False) or Target_2.isFinished:
                    # keep track of stop time/frame for later
                    Target_2.tStop = t  # not accounting for scr refresh
                    Target_2.tStopRefresh = tThisFlipGlobal  # on global time
                    Target_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Target_2.stopped')
                    # update status
                    Target_2.status = FINISHED
                    Target_2.stop()
            
            # *Marsker2_1* updates
            
            # if Marsker2_1 is starting this frame...
            if Marsker2_1.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                Marsker2_1.frameNStart = frameN  # exact frame index
                Marsker2_1.tStart = t  # local t and not account for scr refresh
                Marsker2_1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('Marsker2_1.started', tThisFlipGlobal)
                # update status
                Marsker2_1.status = STARTED
                Marsker2_1.play(when=win)  # sync with win flip
            
            # if Marsker2_1 is stopping this frame...
            if Marsker2_1.status == STARTED:
                if bool(False) or Marsker2_1.isFinished:
                    # keep track of stop time/frame for later
                    Marsker2_1.tStop = t  # not accounting for scr refresh
                    Marsker2_1.tStopRefresh = tThisFlipGlobal  # on global time
                    Marsker2_1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Marsker2_1.stopped')
                    # update status
                    Marsker2_1.status = FINISHED
                    Marsker2_1.stop()
            
            # *Masker2_2* updates
            
            # if Masker2_2 is starting this frame...
            if Masker2_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                Masker2_2.frameNStart = frameN  # exact frame index
                Masker2_2.tStart = t  # local t and not account for scr refresh
                Masker2_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('Masker2_2.started', tThisFlipGlobal)
                # update status
                Masker2_2.status = STARTED
                Masker2_2.play(when=win)  # sync with win flip
            
            # if Masker2_2 is stopping this frame...
            if Masker2_2.status == STARTED:
                if bool(False) or Masker2_2.isFinished:
                    # keep track of stop time/frame for later
                    Masker2_2.tStop = t  # not accounting for scr refresh
                    Masker2_2.tStopRefresh = tThisFlipGlobal  # on global time
                    Masker2_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Masker2_2.stopped')
                    # update status
                    Masker2_2.status = FINISHED
                    Masker2_2.stop()
            
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
                    currentRoutine=bloc_3loc,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                bloc_3loc.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if bloc_3loc.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in bloc_3loc.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "bloc_3loc" ---
        for thisComponent in bloc_3loc.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for bloc_3loc
        bloc_3loc.tStop = globalClock.getTime(format='float')
        bloc_3loc.tStopRefresh = tThisFlipGlobal
        thisExp.addData('bloc_3loc.stopped', bloc_3loc.tStop)
        trials_3loc.addData('R1_3.numClicks', R1_3.numClicks)
        if R1_3.numClicks:
           trials_3loc.addData('R1_3.timesOn', R1_3.timesOn)
           trials_3loc.addData('R1_3.timesOff', R1_3.timesOff)
        else:
           trials_3loc.addData('R1_3.timesOn', "")
           trials_3loc.addData('R1_3.timesOff', "")
        trials_3loc.addData('R2_3.numClicks', R2_3.numClicks)
        if R2_3.numClicks:
           trials_3loc.addData('R2_3.timesOn', R2_3.timesOn)
           trials_3loc.addData('R2_3.timesOff', R2_3.timesOff)
        else:
           trials_3loc.addData('R2_3.timesOn', "")
           trials_3loc.addData('R2_3.timesOff', "")
        trials_3loc.addData('R3_3.numClicks', R3_3.numClicks)
        if R3_3.numClicks:
           trials_3loc.addData('R3_3.timesOn', R3_3.timesOn)
           trials_3loc.addData('R3_3.timesOff', R3_3.timesOff)
        else:
           trials_3loc.addData('R3_3.timesOn', "")
           trials_3loc.addData('R3_3.timesOff', "")
        trials_3loc.addData('R4_3.numClicks', R4_3.numClicks)
        if R4_3.numClicks:
           trials_3loc.addData('R4_3.timesOn', R4_3.timesOn)
           trials_3loc.addData('R4_3.timesOff', R4_3.timesOff)
        else:
           trials_3loc.addData('R4_3.timesOn', "")
           trials_3loc.addData('R4_3.timesOff', "")
        trials_3loc.addData('R5_3.numClicks', R5_3.numClicks)
        if R5_3.numClicks:
           trials_3loc.addData('R5_3.timesOn', R5_3.timesOn)
           trials_3loc.addData('R5_3.timesOff', R5_3.timesOff)
        else:
           trials_3loc.addData('R5_3.timesOn', "")
           trials_3loc.addData('R5_3.timesOff', "")
        trials_3loc.addData('R6_3.numClicks', R6_3.numClicks)
        if R6_3.numClicks:
           trials_3loc.addData('R6_3.timesOn', R6_3.timesOn)
           trials_3loc.addData('R6_3.timesOff', R6_3.timesOff)
        else:
           trials_3loc.addData('R6_3.timesOn', "")
           trials_3loc.addData('R6_3.timesOff', "")
        trials_3loc.addData('R7_3.numClicks', R7_3.numClicks)
        if R7_3.numClicks:
           trials_3loc.addData('R7_3.timesOn', R7_3.timesOn)
           trials_3loc.addData('R7_3.timesOff', R7_3.timesOff)
        else:
           trials_3loc.addData('R7_3.timesOn', "")
           trials_3loc.addData('R7_3.timesOff', "")
        trials_3loc.addData('R8_3.numClicks', R8_3.numClicks)
        if R8_3.numClicks:
           trials_3loc.addData('R8_3.timesOn', R8_3.timesOn)
           trials_3loc.addData('R8_3.timesOff', R8_3.timesOff)
        else:
           trials_3loc.addData('R8_3.timesOn', "")
           trials_3loc.addData('R8_3.timesOff', "")
        trials_3loc.addData('B1_3.numClicks', B1_3.numClicks)
        if B1_3.numClicks:
           trials_3loc.addData('B1_3.timesOn', B1_3.timesOn)
           trials_3loc.addData('B1_3.timesOff', B1_3.timesOff)
        else:
           trials_3loc.addData('B1_3.timesOn', "")
           trials_3loc.addData('B1_3.timesOff', "")
        trials_3loc.addData('B2_3.numClicks', B2_3.numClicks)
        if B2_3.numClicks:
           trials_3loc.addData('B2_3.timesOn', B2_3.timesOn)
           trials_3loc.addData('B2_3.timesOff', B2_3.timesOff)
        else:
           trials_3loc.addData('B2_3.timesOn', "")
           trials_3loc.addData('B2_3.timesOff', "")
        trials_3loc.addData('B3_3.numClicks', B3_3.numClicks)
        if B3_3.numClicks:
           trials_3loc.addData('B3_3.timesOn', B3_3.timesOn)
           trials_3loc.addData('B3_3.timesOff', B3_3.timesOff)
        else:
           trials_3loc.addData('B3_3.timesOn', "")
           trials_3loc.addData('B3_3.timesOff', "")
        trials_3loc.addData('B4_3.numClicks', B4_3.numClicks)
        if B4_3.numClicks:
           trials_3loc.addData('B4_3.timesOn', B4_3.timesOn)
           trials_3loc.addData('B4_3.timesOff', B4_3.timesOff)
        else:
           trials_3loc.addData('B4_3.timesOn', "")
           trials_3loc.addData('B4_3.timesOff', "")
        trials_3loc.addData('B5_3.numClicks', B5_3.numClicks)
        if B5_3.numClicks:
           trials_3loc.addData('B5_3.timesOn', B5_3.timesOn)
           trials_3loc.addData('B5_3.timesOff', B5_3.timesOff)
        else:
           trials_3loc.addData('B5_3.timesOn', "")
           trials_3loc.addData('B5_3.timesOff', "")
        trials_3loc.addData('B6_3.numClicks', B6_3.numClicks)
        if B6_3.numClicks:
           trials_3loc.addData('B6_3.timesOn', B6_3.timesOn)
           trials_3loc.addData('B6_3.timesOff', B6_3.timesOff)
        else:
           trials_3loc.addData('B6_3.timesOn', "")
           trials_3loc.addData('B6_3.timesOff', "")
        trials_3loc.addData('B7_3.numClicks', B7_3.numClicks)
        if B7_3.numClicks:
           trials_3loc.addData('B7_3.timesOn', B7_3.timesOn)
           trials_3loc.addData('B7_3.timesOff', B7_3.timesOff)
        else:
           trials_3loc.addData('B7_3.timesOn', "")
           trials_3loc.addData('B7_3.timesOff', "")
        trials_3loc.addData('B8_3.numClicks', B8_3.numClicks)
        if B8_3.numClicks:
           trials_3loc.addData('B8_3.timesOn', B8_3.timesOn)
           trials_3loc.addData('B8_3.timesOff', B8_3.timesOff)
        else:
           trials_3loc.addData('B8_3.timesOn', "")
           trials_3loc.addData('B8_3.timesOff', "")
        trials_3loc.addData('V1_3.numClicks', V1_3.numClicks)
        if V1_3.numClicks:
           trials_3loc.addData('V1_3.timesOn', V1_3.timesOn)
           trials_3loc.addData('V1_3.timesOff', V1_3.timesOff)
        else:
           trials_3loc.addData('V1_3.timesOn', "")
           trials_3loc.addData('V1_3.timesOff', "")
        trials_3loc.addData('V2_3.numClicks', V2_3.numClicks)
        if V2_3.numClicks:
           trials_3loc.addData('V2_3.timesOn', V2_3.timesOn)
           trials_3loc.addData('V2_3.timesOff', V2_3.timesOff)
        else:
           trials_3loc.addData('V2_3.timesOn', "")
           trials_3loc.addData('V2_3.timesOff', "")
        trials_3loc.addData('V3_3.numClicks', V3_3.numClicks)
        if V3_3.numClicks:
           trials_3loc.addData('V3_3.timesOn', V3_3.timesOn)
           trials_3loc.addData('V3_3.timesOff', V3_3.timesOff)
        else:
           trials_3loc.addData('V3_3.timesOn', "")
           trials_3loc.addData('V3_3.timesOff', "")
        trials_3loc.addData('V4_3.numClicks', V4_3.numClicks)
        if V4_3.numClicks:
           trials_3loc.addData('V4_3.timesOn', V4_3.timesOn)
           trials_3loc.addData('V4_3.timesOff', V4_3.timesOff)
        else:
           trials_3loc.addData('V4_3.timesOn', "")
           trials_3loc.addData('V4_3.timesOff', "")
        trials_3loc.addData('V5_3.numClicks', V5_3.numClicks)
        if V5_3.numClicks:
           trials_3loc.addData('V5_3.timesOn', V5_3.timesOn)
           trials_3loc.addData('V5_3.timesOff', V5_3.timesOff)
        else:
           trials_3loc.addData('V5_3.timesOn', "")
           trials_3loc.addData('V5_3.timesOff', "")
        trials_3loc.addData('V6_3.numClicks', V6_3.numClicks)
        if V6_3.numClicks:
           trials_3loc.addData('V6_3.timesOn', V6_3.timesOn)
           trials_3loc.addData('V6_3.timesOff', V6_3.timesOff)
        else:
           trials_3loc.addData('V6_3.timesOn', "")
           trials_3loc.addData('V6_3.timesOff', "")
        trials_3loc.addData('V7_3.numClicks', V7_3.numClicks)
        if V7_3.numClicks:
           trials_3loc.addData('V7_3.timesOn', V7_3.timesOn)
           trials_3loc.addData('V7_3.timesOff', V7_3.timesOff)
        else:
           trials_3loc.addData('V7_3.timesOn', "")
           trials_3loc.addData('V7_3.timesOff', "")
        trials_3loc.addData('V8_3.numClicks', V8_3.numClicks)
        if V8_3.numClicks:
           trials_3loc.addData('V8_3.timesOn', V8_3.timesOn)
           trials_3loc.addData('V8_3.timesOff', V8_3.timesOff)
        else:
           trials_3loc.addData('V8_3.timesOn', "")
           trials_3loc.addData('V8_3.timesOff', "")
        trials_3loc.addData('J1_3.numClicks', J1_3.numClicks)
        if J1_3.numClicks:
           trials_3loc.addData('J1_3.timesOn', J1_3.timesOn)
           trials_3loc.addData('J1_3.timesOff', J1_3.timesOff)
        else:
           trials_3loc.addData('J1_3.timesOn', "")
           trials_3loc.addData('J1_3.timesOff', "")
        trials_3loc.addData('J2_3.numClicks', J2_3.numClicks)
        if J2_3.numClicks:
           trials_3loc.addData('J2_3.timesOn', J2_3.timesOn)
           trials_3loc.addData('J2_3.timesOff', J2_3.timesOff)
        else:
           trials_3loc.addData('J2_3.timesOn', "")
           trials_3loc.addData('J2_3.timesOff', "")
        trials_3loc.addData('J3_3.numClicks', J3_3.numClicks)
        if J3_3.numClicks:
           trials_3loc.addData('J3_3.timesOn', J3_3.timesOn)
           trials_3loc.addData('J3_3.timesOff', J3_3.timesOff)
        else:
           trials_3loc.addData('J3_3.timesOn', "")
           trials_3loc.addData('J3_3.timesOff', "")
        trials_3loc.addData('J4_3.numClicks', J4_3.numClicks)
        if J4_3.numClicks:
           trials_3loc.addData('J4_3.timesOn', J4_3.timesOn)
           trials_3loc.addData('J4_3.timesOff', J4_3.timesOff)
        else:
           trials_3loc.addData('J4_3.timesOn', "")
           trials_3loc.addData('J4_3.timesOff', "")
        trials_3loc.addData('J5_3.numClicks', J5_3.numClicks)
        if J5_3.numClicks:
           trials_3loc.addData('J5_3.timesOn', J5_3.timesOn)
           trials_3loc.addData('J5_3.timesOff', J5_3.timesOff)
        else:
           trials_3loc.addData('J5_3.timesOn', "")
           trials_3loc.addData('J5_3.timesOff', "")
        trials_3loc.addData('J6_3.numClicks', J6_3.numClicks)
        if J6_3.numClicks:
           trials_3loc.addData('J6_3.timesOn', J6_3.timesOn)
           trials_3loc.addData('J6_3.timesOff', J6_3.timesOff)
        else:
           trials_3loc.addData('J6_3.timesOn', "")
           trials_3loc.addData('J6_3.timesOff', "")
        trials_3loc.addData('J7_3.numClicks', J7_3.numClicks)
        if J7_3.numClicks:
           trials_3loc.addData('J7_3.timesOn', J7_3.timesOn)
           trials_3loc.addData('J7_3.timesOff', J7_3.timesOff)
        else:
           trials_3loc.addData('J7_3.timesOn', "")
           trials_3loc.addData('J7_3.timesOff', "")
        trials_3loc.addData('J8_3.numClicks', J8_3.numClicks)
        if J8_3.numClicks:
           trials_3loc.addData('J8_3.timesOn', J8_3.timesOn)
           trials_3loc.addData('J8_3.timesOff', J8_3.timesOff)
        else:
           trials_3loc.addData('J8_3.timesOn', "")
           trials_3loc.addData('J8_3.timesOff', "")
        Target_2.pause()  # ensure sound has stopped at end of Routine
        Marsker2_1.pause()  # ensure sound has stopped at end of Routine
        Masker2_2.pause()  # ensure sound has stopped at end of Routine
        # the Routine "bloc_3loc" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "escu" ---
        # create an object to store info about Routine escu
        escu = data.Routine(
            name='escu',
            components=[Pasdeffort, __1, Trespeudeffort, __2, Peudeffort, __3, Effortmodere, __4, Effortconsiderable, __5, Beaucoupdeffort, Quedubruit],
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
        # reset __2 to account for continued clicks & clear times on/off
        __2.reset()
        # reset Peudeffort to account for continued clicks & clear times on/off
        Peudeffort.reset()
        # reset __3 to account for continued clicks & clear times on/off
        __3.reset()
        # reset Effortmodere to account for continued clicks & clear times on/off
        Effortmodere.reset()
        # reset __4 to account for continued clicks & clear times on/off
        __4.reset()
        # reset Effortconsiderable to account for continued clicks & clear times on/off
        Effortconsiderable.reset()
        # reset __5 to account for continued clicks & clear times on/off
        __5.reset()
        # reset Beaucoupdeffort to account for continued clicks & clear times on/off
        Beaucoupdeffort.reset()
        # reset Quedubruit to account for continued clicks & clear times on/off
        Quedubruit.reset()
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
            # if trial has changed, end Routine now
            if hasattr(thisTrials_3loc, 'status') and thisTrials_3loc.status == STOPPING:
                continueRoutine = False
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
            # *__2* updates
            
            # if __2 is starting this frame...
            if __2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                __2.frameNStart = frameN  # exact frame index
                __2.tStart = t  # local t and not account for scr refresh
                __2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(__2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, '__2.started')
                # update status
                __2.status = STARTED
                win.callOnFlip(__2.buttonClock.reset)
                __2.setAutoDraw(True)
            
            # if __2 is active this frame...
            if __2.status == STARTED:
                # update params
                pass
                # check whether __2 has been pressed
                if __2.isClicked:
                    if not __2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        __2.timesOn.append(__2.buttonClock.getTime())
                        __2.timesOff.append(__2.buttonClock.getTime())
                    elif len(__2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        __2.timesOff[-1] = __2.buttonClock.getTime()
                    if not __2.wasClicked:
                        # end routine when __2 is clicked
                        continueRoutine = False
                    if not __2.wasClicked:
                        # run callback code when __2 is clicked
                        pass
            # take note of whether __2 was clicked, so that next frame we know if clicks are new
            __2.wasClicked = __2.isClicked and __2.status == STARTED
            # *Peudeffort* updates
            
            # if Peudeffort is starting this frame...
            if Peudeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Peudeffort.frameNStart = frameN  # exact frame index
                Peudeffort.tStart = t  # local t and not account for scr refresh
                Peudeffort.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Peudeffort, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Peudeffort.started')
                # update status
                Peudeffort.status = STARTED
                win.callOnFlip(Peudeffort.buttonClock.reset)
                Peudeffort.setAutoDraw(True)
            
            # if Peudeffort is active this frame...
            if Peudeffort.status == STARTED:
                # update params
                pass
                # check whether Peudeffort has been pressed
                if Peudeffort.isClicked:
                    if not Peudeffort.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Peudeffort.timesOn.append(Peudeffort.buttonClock.getTime())
                        Peudeffort.timesOff.append(Peudeffort.buttonClock.getTime())
                    elif len(Peudeffort.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Peudeffort.timesOff[-1] = Peudeffort.buttonClock.getTime()
                    if not Peudeffort.wasClicked:
                        # end routine when Peudeffort is clicked
                        continueRoutine = False
                    if not Peudeffort.wasClicked:
                        # run callback code when Peudeffort is clicked
                        pass
            # take note of whether Peudeffort was clicked, so that next frame we know if clicks are new
            Peudeffort.wasClicked = Peudeffort.isClicked and Peudeffort.status == STARTED
            # *__3* updates
            
            # if __3 is starting this frame...
            if __3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                __3.frameNStart = frameN  # exact frame index
                __3.tStart = t  # local t and not account for scr refresh
                __3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(__3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, '__3.started')
                # update status
                __3.status = STARTED
                win.callOnFlip(__3.buttonClock.reset)
                __3.setAutoDraw(True)
            
            # if __3 is active this frame...
            if __3.status == STARTED:
                # update params
                pass
                # check whether __3 has been pressed
                if __3.isClicked:
                    if not __3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        __3.timesOn.append(__3.buttonClock.getTime())
                        __3.timesOff.append(__3.buttonClock.getTime())
                    elif len(__3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        __3.timesOff[-1] = __3.buttonClock.getTime()
                    if not __3.wasClicked:
                        # end routine when __3 is clicked
                        continueRoutine = False
                    if not __3.wasClicked:
                        # run callback code when __3 is clicked
                        pass
            # take note of whether __3 was clicked, so that next frame we know if clicks are new
            __3.wasClicked = __3.isClicked and __3.status == STARTED
            # *Effortmodere* updates
            
            # if Effortmodere is starting this frame...
            if Effortmodere.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Effortmodere.frameNStart = frameN  # exact frame index
                Effortmodere.tStart = t  # local t and not account for scr refresh
                Effortmodere.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Effortmodere, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Effortmodere.started')
                # update status
                Effortmodere.status = STARTED
                win.callOnFlip(Effortmodere.buttonClock.reset)
                Effortmodere.setAutoDraw(True)
            
            # if Effortmodere is active this frame...
            if Effortmodere.status == STARTED:
                # update params
                pass
                # check whether Effortmodere has been pressed
                if Effortmodere.isClicked:
                    if not Effortmodere.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Effortmodere.timesOn.append(Effortmodere.buttonClock.getTime())
                        Effortmodere.timesOff.append(Effortmodere.buttonClock.getTime())
                    elif len(Effortmodere.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Effortmodere.timesOff[-1] = Effortmodere.buttonClock.getTime()
                    if not Effortmodere.wasClicked:
                        # end routine when Effortmodere is clicked
                        continueRoutine = False
                    if not Effortmodere.wasClicked:
                        # run callback code when Effortmodere is clicked
                        pass
            # take note of whether Effortmodere was clicked, so that next frame we know if clicks are new
            Effortmodere.wasClicked = Effortmodere.isClicked and Effortmodere.status == STARTED
            # *__4* updates
            
            # if __4 is starting this frame...
            if __4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                __4.frameNStart = frameN  # exact frame index
                __4.tStart = t  # local t and not account for scr refresh
                __4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(__4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, '__4.started')
                # update status
                __4.status = STARTED
                win.callOnFlip(__4.buttonClock.reset)
                __4.setAutoDraw(True)
            
            # if __4 is active this frame...
            if __4.status == STARTED:
                # update params
                pass
                # check whether __4 has been pressed
                if __4.isClicked:
                    if not __4.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        __4.timesOn.append(__4.buttonClock.getTime())
                        __4.timesOff.append(__4.buttonClock.getTime())
                    elif len(__4.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        __4.timesOff[-1] = __4.buttonClock.getTime()
                    if not __4.wasClicked:
                        # end routine when __4 is clicked
                        continueRoutine = False
                    if not __4.wasClicked:
                        # run callback code when __4 is clicked
                        pass
            # take note of whether __4 was clicked, so that next frame we know if clicks are new
            __4.wasClicked = __4.isClicked and __4.status == STARTED
            # *Effortconsiderable* updates
            
            # if Effortconsiderable is starting this frame...
            if Effortconsiderable.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Effortconsiderable.frameNStart = frameN  # exact frame index
                Effortconsiderable.tStart = t  # local t and not account for scr refresh
                Effortconsiderable.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Effortconsiderable, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Effortconsiderable.started')
                # update status
                Effortconsiderable.status = STARTED
                win.callOnFlip(Effortconsiderable.buttonClock.reset)
                Effortconsiderable.setAutoDraw(True)
            
            # if Effortconsiderable is active this frame...
            if Effortconsiderable.status == STARTED:
                # update params
                pass
                # check whether Effortconsiderable has been pressed
                if Effortconsiderable.isClicked:
                    if not Effortconsiderable.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Effortconsiderable.timesOn.append(Effortconsiderable.buttonClock.getTime())
                        Effortconsiderable.timesOff.append(Effortconsiderable.buttonClock.getTime())
                    elif len(Effortconsiderable.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Effortconsiderable.timesOff[-1] = Effortconsiderable.buttonClock.getTime()
                    if not Effortconsiderable.wasClicked:
                        # end routine when Effortconsiderable is clicked
                        continueRoutine = False
                    if not Effortconsiderable.wasClicked:
                        # run callback code when Effortconsiderable is clicked
                        pass
            # take note of whether Effortconsiderable was clicked, so that next frame we know if clicks are new
            Effortconsiderable.wasClicked = Effortconsiderable.isClicked and Effortconsiderable.status == STARTED
            # *__5* updates
            
            # if __5 is starting this frame...
            if __5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                __5.frameNStart = frameN  # exact frame index
                __5.tStart = t  # local t and not account for scr refresh
                __5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(__5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, '__5.started')
                # update status
                __5.status = STARTED
                win.callOnFlip(__5.buttonClock.reset)
                __5.setAutoDraw(True)
            
            # if __5 is active this frame...
            if __5.status == STARTED:
                # update params
                pass
                # check whether __5 has been pressed
                if __5.isClicked:
                    if not __5.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        __5.timesOn.append(__5.buttonClock.getTime())
                        __5.timesOff.append(__5.buttonClock.getTime())
                    elif len(__5.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        __5.timesOff[-1] = __5.buttonClock.getTime()
                    if not __5.wasClicked:
                        # end routine when __5 is clicked
                        continueRoutine = False
                    if not __5.wasClicked:
                        # run callback code when __5 is clicked
                        pass
            # take note of whether __5 was clicked, so that next frame we know if clicks are new
            __5.wasClicked = __5.isClicked and __5.status == STARTED
            # *Beaucoupdeffort* updates
            
            # if Beaucoupdeffort is starting this frame...
            if Beaucoupdeffort.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Beaucoupdeffort.frameNStart = frameN  # exact frame index
                Beaucoupdeffort.tStart = t  # local t and not account for scr refresh
                Beaucoupdeffort.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Beaucoupdeffort, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Beaucoupdeffort.started')
                # update status
                Beaucoupdeffort.status = STARTED
                win.callOnFlip(Beaucoupdeffort.buttonClock.reset)
                Beaucoupdeffort.setAutoDraw(True)
            
            # if Beaucoupdeffort is active this frame...
            if Beaucoupdeffort.status == STARTED:
                # update params
                pass
                # check whether Beaucoupdeffort has been pressed
                if Beaucoupdeffort.isClicked:
                    if not Beaucoupdeffort.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Beaucoupdeffort.timesOn.append(Beaucoupdeffort.buttonClock.getTime())
                        Beaucoupdeffort.timesOff.append(Beaucoupdeffort.buttonClock.getTime())
                    elif len(Beaucoupdeffort.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Beaucoupdeffort.timesOff[-1] = Beaucoupdeffort.buttonClock.getTime()
                    if not Beaucoupdeffort.wasClicked:
                        # end routine when Beaucoupdeffort is clicked
                        continueRoutine = False
                    if not Beaucoupdeffort.wasClicked:
                        # run callback code when Beaucoupdeffort is clicked
                        pass
            # take note of whether Beaucoupdeffort was clicked, so that next frame we know if clicks are new
            Beaucoupdeffort.wasClicked = Beaucoupdeffort.isClicked and Beaucoupdeffort.status == STARTED
            # *Quedubruit* updates
            
            # if Quedubruit is starting this frame...
            if Quedubruit.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Quedubruit.frameNStart = frameN  # exact frame index
                Quedubruit.tStart = t  # local t and not account for scr refresh
                Quedubruit.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Quedubruit, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Quedubruit.started')
                # update status
                Quedubruit.status = STARTED
                win.callOnFlip(Quedubruit.buttonClock.reset)
                Quedubruit.setAutoDraw(True)
            
            # if Quedubruit is active this frame...
            if Quedubruit.status == STARTED:
                # update params
                pass
                # check whether Quedubruit has been pressed
                if Quedubruit.isClicked:
                    if not Quedubruit.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        Quedubruit.timesOn.append(Quedubruit.buttonClock.getTime())
                        Quedubruit.timesOff.append(Quedubruit.buttonClock.getTime())
                    elif len(Quedubruit.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        Quedubruit.timesOff[-1] = Quedubruit.buttonClock.getTime()
                    if not Quedubruit.wasClicked:
                        # end routine when Quedubruit is clicked
                        continueRoutine = False
                    if not Quedubruit.wasClicked:
                        # run callback code when Quedubruit is clicked
                        pass
            # take note of whether Quedubruit was clicked, so that next frame we know if clicks are new
            Quedubruit.wasClicked = Quedubruit.isClicked and Quedubruit.status == STARTED
            
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
        trials_3loc.addData('Pasdeffort.numClicks', Pasdeffort.numClicks)
        if Pasdeffort.numClicks:
           trials_3loc.addData('Pasdeffort.timesOn', Pasdeffort.timesOn)
           trials_3loc.addData('Pasdeffort.timesOff', Pasdeffort.timesOff)
        else:
           trials_3loc.addData('Pasdeffort.timesOn', "")
           trials_3loc.addData('Pasdeffort.timesOff', "")
        trials_3loc.addData('__1.numClicks', __1.numClicks)
        if __1.numClicks:
           trials_3loc.addData('__1.timesOn', __1.timesOn)
           trials_3loc.addData('__1.timesOff', __1.timesOff)
        else:
           trials_3loc.addData('__1.timesOn', "")
           trials_3loc.addData('__1.timesOff', "")
        trials_3loc.addData('Trespeudeffort.numClicks', Trespeudeffort.numClicks)
        if Trespeudeffort.numClicks:
           trials_3loc.addData('Trespeudeffort.timesOn', Trespeudeffort.timesOn)
           trials_3loc.addData('Trespeudeffort.timesOff', Trespeudeffort.timesOff)
        else:
           trials_3loc.addData('Trespeudeffort.timesOn', "")
           trials_3loc.addData('Trespeudeffort.timesOff', "")
        trials_3loc.addData('__2.numClicks', __2.numClicks)
        if __2.numClicks:
           trials_3loc.addData('__2.timesOn', __2.timesOn)
           trials_3loc.addData('__2.timesOff', __2.timesOff)
        else:
           trials_3loc.addData('__2.timesOn', "")
           trials_3loc.addData('__2.timesOff', "")
        trials_3loc.addData('Peudeffort.numClicks', Peudeffort.numClicks)
        if Peudeffort.numClicks:
           trials_3loc.addData('Peudeffort.timesOn', Peudeffort.timesOn)
           trials_3loc.addData('Peudeffort.timesOff', Peudeffort.timesOff)
        else:
           trials_3loc.addData('Peudeffort.timesOn', "")
           trials_3loc.addData('Peudeffort.timesOff', "")
        trials_3loc.addData('__3.numClicks', __3.numClicks)
        if __3.numClicks:
           trials_3loc.addData('__3.timesOn', __3.timesOn)
           trials_3loc.addData('__3.timesOff', __3.timesOff)
        else:
           trials_3loc.addData('__3.timesOn', "")
           trials_3loc.addData('__3.timesOff', "")
        trials_3loc.addData('Effortmodere.numClicks', Effortmodere.numClicks)
        if Effortmodere.numClicks:
           trials_3loc.addData('Effortmodere.timesOn', Effortmodere.timesOn)
           trials_3loc.addData('Effortmodere.timesOff', Effortmodere.timesOff)
        else:
           trials_3loc.addData('Effortmodere.timesOn', "")
           trials_3loc.addData('Effortmodere.timesOff', "")
        trials_3loc.addData('__4.numClicks', __4.numClicks)
        if __4.numClicks:
           trials_3loc.addData('__4.timesOn', __4.timesOn)
           trials_3loc.addData('__4.timesOff', __4.timesOff)
        else:
           trials_3loc.addData('__4.timesOn', "")
           trials_3loc.addData('__4.timesOff', "")
        trials_3loc.addData('Effortconsiderable.numClicks', Effortconsiderable.numClicks)
        if Effortconsiderable.numClicks:
           trials_3loc.addData('Effortconsiderable.timesOn', Effortconsiderable.timesOn)
           trials_3loc.addData('Effortconsiderable.timesOff', Effortconsiderable.timesOff)
        else:
           trials_3loc.addData('Effortconsiderable.timesOn', "")
           trials_3loc.addData('Effortconsiderable.timesOff', "")
        trials_3loc.addData('__5.numClicks', __5.numClicks)
        if __5.numClicks:
           trials_3loc.addData('__5.timesOn', __5.timesOn)
           trials_3loc.addData('__5.timesOff', __5.timesOff)
        else:
           trials_3loc.addData('__5.timesOn', "")
           trials_3loc.addData('__5.timesOff', "")
        trials_3loc.addData('Beaucoupdeffort.numClicks', Beaucoupdeffort.numClicks)
        if Beaucoupdeffort.numClicks:
           trials_3loc.addData('Beaucoupdeffort.timesOn', Beaucoupdeffort.timesOn)
           trials_3loc.addData('Beaucoupdeffort.timesOff', Beaucoupdeffort.timesOff)
        else:
           trials_3loc.addData('Beaucoupdeffort.timesOn', "")
           trials_3loc.addData('Beaucoupdeffort.timesOff', "")
        trials_3loc.addData('Quedubruit.numClicks', Quedubruit.numClicks)
        if Quedubruit.numClicks:
           trials_3loc.addData('Quedubruit.timesOn', Quedubruit.timesOn)
           trials_3loc.addData('Quedubruit.timesOff', Quedubruit.timesOff)
        else:
           trials_3loc.addData('Quedubruit.timesOn', "")
           trials_3loc.addData('Quedubruit.timesOff', "")
        # the Routine "escu" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials_3loc as finished
        if hasattr(thisTrials_3loc, 'status'):
            thisTrials_3loc.status = FINISHED
        # if awaiting a pause, pause now
        if trials_3loc.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_3loc.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'trials_3loc'
    trials_3loc.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "End" ---
    # create an object to store info about Routine End
    End = data.Routine(
        name='End',
        components=[EndButton, Endtext],
    )
    End.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset EndButton to account for continued clicks & clear times on/off
    EndButton.reset()
    # store start times for End
    End.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End.tStart = globalClock.getTime(format='float')
    End.status = STARTED
    thisExp.addData('End.started', End.tStart)
    End.maxDuration = None
    # keep track of which components have finished
    EndComponents = End.components
    for thisComponent in End.components:
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
    
    # --- Run Routine "End" ---
    thisExp.currentRoutine = End
    End.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *EndButton* updates
        
        # if EndButton is starting this frame...
        if EndButton.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            EndButton.frameNStart = frameN  # exact frame index
            EndButton.tStart = t  # local t and not account for scr refresh
            EndButton.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(EndButton, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'EndButton.started')
            # update status
            EndButton.status = STARTED
            win.callOnFlip(EndButton.buttonClock.reset)
            EndButton.setAutoDraw(True)
        
        # if EndButton is active this frame...
        if EndButton.status == STARTED:
            # update params
            pass
            # check whether EndButton has been pressed
            if EndButton.isClicked:
                if not EndButton.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    EndButton.timesOn.append(EndButton.buttonClock.getTime())
                    EndButton.timesOff.append(EndButton.buttonClock.getTime())
                elif len(EndButton.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    EndButton.timesOff[-1] = EndButton.buttonClock.getTime()
                if not EndButton.wasClicked:
                    # end routine when EndButton is clicked
                    continueRoutine = False
                if not EndButton.wasClicked:
                    # run callback code when EndButton is clicked
                    pass
        # take note of whether EndButton was clicked, so that next frame we know if clicks are new
        EndButton.wasClicked = EndButton.isClicked and EndButton.status == STARTED
        
        # *Endtext* updates
        
        # if Endtext is starting this frame...
        if Endtext.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Endtext.frameNStart = frameN  # exact frame index
            Endtext.tStart = t  # local t and not account for scr refresh
            Endtext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Endtext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Endtext.started')
            # update status
            Endtext.status = STARTED
            Endtext.setAutoDraw(True)
        
        # if Endtext is active this frame...
        if Endtext.status == STARTED:
            # update params
            pass
        
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
                currentRoutine=End,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            End.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if End.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in End.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in End.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End
    End.tStop = globalClock.getTime(format='float')
    End.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End.stopped', End.tStop)
    thisExp.addData('EndButton.numClicks', EndButton.numClicks)
    if EndButton.numClicks:
       thisExp.addData('EndButton.timesOn', EndButton.timesOn)
       thisExp.addData('EndButton.timesOff', EndButton.timesOff)
    else:
       thisExp.addData('EndButton.timesOn', "")
       thisExp.addData('EndButton.timesOff', "")
    thisExp.nextEntry()
    # the Routine "End" was not non-slip safe, so reset the non-slip timer
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
