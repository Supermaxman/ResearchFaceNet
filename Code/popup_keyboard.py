# MIT License
# 
# Copyright (c) 2018 Pete Mojeiko
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from https://github.com/petemojeiko/tkinter-keyboard/blob/master/keyboard.py modified for python3 and 
# for better interactions on raspberry pi touch screen

import tkinter as tk

class _PopupKeyboard(tk.Toplevel):
    '''A Toplevel instance that displays a keyboard that is attached to
    another widget. Only the Entry widget has a subclass in this version.
    '''
    
    def __init__(self, parent, attach, x, y, keycolor, keysize=5, font=None):
        tk.Toplevel.__init__(self, takefocus=0)
        
        self.overrideredirect(True)
        self.attributes('-alpha', 0.85)

        self.parent = parent
        self.attach = attach
        self.keysize = keysize
        self.keycolor = keycolor
        self.x = x
        self.y = y
        self.font = font

        self.row1 = tk.Frame(self)
        self.row2 = tk.Frame(self)
        self.row3 = tk.Frame(self)
        self.row4 = tk.Frame(self)

        self.row1.grid(row=1)
        self.row2.grid(row=2)
        self.row3.grid(row=3)
        self.row4.grid(row=4)
        
        self._init_keys()

        # destroy _PopupKeyboard on keyboard interrupt
        self.bind('<Key>', lambda e: self._destroy_popup())
        # resize to fit keys
        self.update_idletasks()
        self.geometry('{}x{}+{}+{}'.format(self.winfo_width(),
                                           self.winfo_height(),
                                           self.x, self.y))
        self.shifted = False

    def _init_keys(self):
        self.alpha = {
            'row1' : ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p','/'],
            'row2' : ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l','.'],
            'row3' : ['shift','z', 'x', 'c', 'v', 'b', 'n', 'm',','],
            'row4' : ['[ space ]']
            }
        self.row_offset = {
                'row1' : 1,
                'row2' : 2,
                'row3' : 2,
                'row4' : 5
            }
        self.row_parent = {
                'row1' : self.row1,
                'row2' : self.row2,
                'row3' : self.row3,
                'row4' : self.row4
            }
        self.k_vars = []

        for row, keys in self.alpha.items():
            i = self.row_offset[row]
            p_row = self.row_parent[row]
            
            for k in keys:
                p_width = self.keysize
                p_color = self.keycolor
                if k == '[ space ]':
                    p_width = self.keysize * 3
                k_var = tk.StringVar(p_row, value=k)
                self.k_vars.append(k_var)
                if self.font != None:
                    btn = tk.Button(p_row,
                            textvariable=k_var,
                            width=p_width,
                            bg=p_color,
                            font=self.font,
                            command=lambda k_var=k_var: self._attach_key_press(k_var.get()))
                else:
                    btn = tk.Button(p_row,
                            textvariable=k_var,
                            width=p_width,
                            bg=p_color,
                            command=lambda k_var=k_var: self._attach_key_press(k_var.get()))
                btn.grid(row=0, column=i)
                i += 1

    def _destroy_popup(self):
        self.destroy()

    def _attach_key_press(self, k):
        k_lower = k.lower()
        prev_shifted = self.shifted
        if k_lower == '>>>':
            self.attach.tk_focusNext().focus_set()
            self.destroy()
        elif k_lower == '<<<':
            self.attach.tk_focusPrev().focus_set()
            self.destroy()
        elif k_lower == '[1,2,3]':
            pass
        elif k_lower == '[ space ]':
            self.attach.insert(tk.END, ' ')
        elif k_lower == 'shift':
            self.shifted = not self.shifted
        else:
            self.attach.insert(tk.END, k)
            if self.shifted:
                self.shifted = False
        if prev_shifted != self.shifted:
            for k_var in self.k_vars:
                prev_value = k_var.get()
                k_var_lower = prev_value.lower()
                if k_var_lower == '[ space ]':
                    continue
                k_var.set(prev_value.upper() if self.shifted else prev_value.lower())
'''
TO-DO: Implement Number Pad
class _PopupNumpad(tk.Toplevel):
    def __init__(self, x, y, keycolor='gray', keysize=5):
        tk.Toplevel.__init__(self, takefocus=0)
        
        self.overrideredirect(True)
        self.attributes('-alpha',0.85)
        self.numframe = Frame(self)
        self.numframe.grid(row=1, column=1)
        self.__init_nums()
        self.update_idletasks()
        self.geometry('{}x{}+{}+{}'.format(self.winfo_width(),
                                           self.winfo_height(),
                                           self.x,self.y))
    def __init_nums(self):
        i=0
        for num in ['7','4','1','8','5','2','9','6','3']:
            tk.Button(self.numframe,
                   text=num,
                   width=int(self.keysize/2),
                   bg=self.keycolor,
                   command=lambda num=num: self.__attach_key_press(num)).grid(row=i%3, column=i/3)
            i+=1
'''

class KeyboardEntry(tk.Frame):
    '''An extension/subclass of the Tkinter Entry widget, capable
    of accepting all existing args, plus a keysize and keycolor option.
    Will pop up an instance of _PopupKeyboard when focus moves into
    the widget
    Usage:
    KeyboardEntry(parent, keysize=6, keycolor='white').pack()
    '''
    
    def __init__(self, parent, keysize=5, keycolor='gray', yoffset=10, *args, **kwargs):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        self.entry = tk.Entry(self, *args, **kwargs)
        self.entry.pack()

        self.keysize = keysize
        self.keycolor = keycolor
        self.yoffset = yoffset
        self.font=None if 'font' not in kwargs else kwargs['font']
        
        self.state = 'idle'
        
        self.entry.bind('<FocusIn>', lambda e: self._check_state('focusin'))
        self.entry.bind('<FocusOut>', lambda e: self._check_state('focusout'))
        self.entry.bind('<Key>', lambda e: self._check_state('keypress'))
        self.kb = None

    def get(self):
        return self.entry.get()

    def _check_state(self, event):
        '''finite state machine'''
        if self.state == 'idle':
            if event == 'focusin':
                self._call_popup()
                self.state = 'virtualkeyboard'
        elif self.state == 'virtualkeyboard':
            if event == 'focusin':
                self._destroy_popup()
                self.state = 'typing'
            elif event == 'keypress':
                self._destroy_popup()
                self.state = 'typing'
        elif self.state == 'typing':
            if event == 'focusout':
                self.state = 'idle'
        
    def _call_popup(self):
        self.kb = _PopupKeyboard(attach=self.entry,
                                 parent=self.parent,
                                 x=self.entry.winfo_rootx(),
                                 y=self.entry.winfo_rooty() + self.entry.winfo_reqheight() + self.yoffset,
                                 keysize=self.keysize,
                                 keycolor=self.keycolor,
                                 font=self.font)

    def destroy(self):
        self._destroy_popup()
        return super().destroy()

    def _destroy_popup(self):
        if self.kb != None:
            self.kb._destroy_popup()