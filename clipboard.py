import win32api, win32clipboard
import win32con
win32clipboard.OpenClipboard()
win32clipboard.EmptyClipboard()
test = "SELAMKRDŞ"
win32clipboard.SetClipboardText(test, win32con.CF_TEXT)
win32clipboard.CloseClipboard()