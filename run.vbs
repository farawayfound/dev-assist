' ExoCortex silent launcher.
' Double-click this (or make a shortcut to it) to start ExoCortex with NO console
' window and NO taskbar entry for the launcher itself. The overlay is the only UI;
' it runs under pythonw.exe (the windowless Python host), so nothing shows in the
' taskbar or Alt-Tab except the overlay, which is itself capture-excluded.
'
' Setup runs automatically whenever it's needed (no .venv yet, the dependency stamp is missing,
' or requirements.txt/run.bat/the detected GPU vendor changed since the stamp was written --
' run.bat itself re-checks all of that). Only a genuinely fresh install (no .venv at all) shows a
' normal, visible console for it, so the very first run can be watched; an existing, already-
' working install (every currently-deployed machine included) never sees a console just because
' it's catching up on a stamp migration or a dependency/vendor change -- that setup still runs for
' real, just with its window hidden, so this launcher looks silent even on that one-time refresh.

Option Explicit

Dim fso, shell, here, pyw, pyExe, bat, stamp
Dim fresh, broken, needsSetup, stampTime, ready

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

here  = fso.GetParentFolderName(WScript.ScriptFullName)
pyw   = here & "\.venv\Scripts\pythonw.exe"
pyExe = here & "\.venv\Scripts\python.exe"
bat   = here & "\run.bat"
stamp = here & "\.venv\exocortex-setup.stamp"

shell.CurrentDirectory = here

fresh = Not fso.FileExists(pyw)
' A .venv whose Python was deleted out from under it (see :create_venv in run.bat) still has its
' pythonw.exe on disk -- which then fails to start without a word, and so did this launcher. It is
' still an existing install, so its repair runs hidden like any other catch-up below.
broken = False
If Not fresh Then broken = Not InterpreterStarts()
needsSetup = fresh Or broken

If Not needsSetup Then
    If Not fso.FileExists(stamp) Then
        ' A working .venv from before the stamp existed (or one whose stamp was removed by hand):
        ' treat it as needing a one-time migration setup, run silently (see IsNewerThanStamp).
        needsSetup = True
    Else
        stampTime = fso.GetFile(stamp).DateLastModified
        needsSetup = IsNewerThanStamp(here & "\requirements.txt", stampTime) _
                  Or IsNewerThanStamp(here & "\requirements-nvidia.txt", stampTime) _
                  Or IsNewerThanStamp(here & "\requirements-directml.txt", stampTime) _
                  Or IsNewerThanStamp(bat, stampTime)
        If Not needsSetup Then
            ' Same vendor recheck run.bat's own :ensure_deps does: a laptop's detected primary GPU
            ' can change (driver update, hybrid-graphics/Optimus switch, dock change) without any
            ' watched file's mtime moving. Skip it (fail open, no setup) only if the probe itself
            ' couldn't run; an empty/unreadable stamp vendor is treated as a mismatch.
            Dim curVendor
            curVendor = DetectedVendor()
            If Len(curVendor) > 0 And curVendor <> StampVendor(stamp) Then needsSetup = True
        End If
    End If
End If

If needsSetup Then
    If fresh Then
        ' Genuinely fresh install: no .venv yet, nothing to preserve the silence of. Visible,
        ' blocking window so the one-time install can be watched -- and so run.bat's first-run
        ' step (`python -m exocortex.setup`) can ask which LLM to pull.
        shell.Run "cmd /c """ & bat & """ --setup-only", 1, True
    Else
        ' An existing install just catching up (stamp migration, a changed requirements*.txt/
        ' run.bat, a changed GPU vendor, or a repair of its deleted Python): still do the real
        ' setup work and still wait for it, but never flash a console for it. EXO_SETUP_QUIET=1
        ' tells run.bat's first-run step to skip its interactive "pull a model" prompt: nobody
        ' could answer it in a hidden window, and the app offers its Models window instead. It
        ' also keeps run.bat from deleting a broken .venv to rebuild it (see :create_venv there).
        shell.Environment("PROCESS")("EXO_SETUP_QUIET") = "1"
        shell.Run "cmd /c """ & bat & """ --setup-only", 0, True
    End If
End If

' A repair that did not take leaves pythonw.exe in place, still unable to start. Launching it would
' do nothing, and the visible run.bat below is a fresh install's fallback -- an existing install
' never shows a console, not even to explain a failure. So: nothing, until run.bat itself is run.
ready = fso.FileExists(pyw)
If ready And broken Then ready = InterpreterStarts()
If ready Then
    ' 0 = hidden window, False = don't wait. pythonw has no console at all.
    ' --replace: force-close any ExoCortex already running (and its model server) first,
    ' so this launch is always the only session (same as run.bat).
    shell.Run """" & pyw & """ -m exocortex --replace", 0, False
ElseIf Not broken Then
    shell.Run "cmd /c """ & bat & """", 1, False
End If

Function IsNewerThanStamp(path, stampTime)
    If fso.FileExists(path) Then
        IsNewerThanStamp = (fso.GetFile(path).DateLastModified > stampTime)
    Else
        IsNewerThanStamp = False
    End If
End Function

' True when the venv's python.exe actually starts -- the same check as :create_venv in run.bat,
' including -E (a stray PYTHONHOME or PYTHONPATH is not the venv's fault). Hidden window, for the
' reason DetectedVendor gives below.
Function InterpreterStarts()
    Dim rc
    InterpreterStarts = False
    If Not fso.FileExists(pyExe) Then Exit Function
    On Error Resume Next
    rc = shell.Run("""" & pyExe & """ -E -c ""import sys""", 0, True)
    If Err.Number = 0 And rc = 0 Then InterpreterStarts = True
    Err.Clear
    On Error Goto 0
End Function

' exocortex.gpu is stdlib-only (ctypes), so this costs a few ms, same as run.bat's own probe.
' Returns "" (never forces setup on its own) if the venv's python.exe can't be run for any reason.
' Runs through shell.Run with a HIDDEN window and the output redirected to a temp file: shell.Exec
' would flash a black console for the console-mode python.exe on every launch, and this launcher
' exists precisely so that nothing but the overlay ever appears (README: "completely windowless").
Function DetectedVendor()
    Dim outFile, ts, out, rc
    DetectedVendor = ""
    If Not fso.FileExists(pyExe) Then Exit Function
    out = ""
    On Error Resume Next
    outFile = fso.BuildPath(fso.GetSpecialFolder(2), "exocortex-vendor-" & fso.GetTempName())
    rc = shell.Run("cmd /c """"" & pyExe & """ -m exocortex.gpu --vendor > """ & outFile & """""", 0, True)
    If Err.Number = 0 And rc = 0 And fso.FileExists(outFile) Then
        Set ts = fso.OpenTextFile(outFile, 1)
        If Err.Number = 0 Then
            If Not ts.AtEndOfStream Then out = ts.ReadAll()
            ts.Close
        End If
    End If
    If fso.FileExists(outFile) Then fso.DeleteFile outFile, True
    ' VBScript's Trim only strips spaces, not the trailing CR/LF a subprocess's stdout ends with --
    ' strip those explicitly, since "--vendor" prints a single bare word.
    out = Trim(Replace(Replace(out, vbCr, ""), vbLf, ""))
    If Err.Number <> 0 Then out = ""
    On Error Goto 0
    DetectedVendor = out
End Function

Function StampVendor(path)
    Dim ts, line
    StampVendor = ""
    On Error Resume Next
    Set ts = fso.OpenTextFile(path, 1)
    If Err.Number <> 0 Then
        Err.Clear
        Exit Function
    End If
    line = Trim(ts.ReadLine())
    ts.Close
    On Error Goto 0
    If Len(line) = 0 Then Exit Function
    StampVendor = Split(line)(0)
End Function
