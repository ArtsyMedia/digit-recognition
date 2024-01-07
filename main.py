import neural_network

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as poop
from kivy.graphics import Color, Ellipse, Line
from kivy.core.window import Window
from kivy.clock import Clock

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

Config.set('graphics', 'resizable', 0)
Config.write()

draw_width = 50

def ProccessImg():
    im = Image.open("images/drawing.png")
    im = ImageOps.grayscale(im)
    im = ImageOps.contain(im, (8,8))
    im.save("images/drawing.png")

    rawData = im.load()
    data = []
    for y in range(8):
        for x in range(8):
            data.append(rawData[x,y])

    confidences = neural_network.Predict(np.asarray(data))

    names = list(confidences.keys())
    values = list(confidences.values())


    plt.clf()
    plt.xticks(range(10))
    plt.yticks(range(0, 100, 5))
    plt.bar(names, values)
    plt.savefig("images/graph.png")



class DrawInput(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1)
            d = draw_width
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y))
        self.export_to_png("images/drawing.png")
        ProccessImg()

    def on_touch_move(self, touch):
        Color(1, 1, 0)
        touch.ud["line"].width = draw_width
        touch.ud["line"].points += (touch.x, touch.y)
        self.export_to_png("images/drawing.png")
        ProccessImg()

    def on_touch_up(self, touch):
        print("RELEASED!", touch)
        self.export_to_png("images/drawing.png")
        ProccessImg()


class SimpleKivy4(App):
    def build(self):
        Window.size = (1000, 500)

        root = FloatLayout(size=(400, 200))
        layout = BoxLayout(orientation="horizontal", size=(
            200, 100), size_hint=(None, None))

        self.painter = DrawInput()
        self.painter.size_hint = (0.5, 1)
        layout.add_widget(self.painter)

        clearBtn = Button(text="Clear")
        clearBtn.size_hint = (0.1, 0.1)
        clearBtn.bind(on_release=self.clear_canvas)
        root.add_widget(clearBtn)

        b = poop(source="images/graph.png", allow_stretch=True, keep_ratio=False)
        b.size_hint = (0.5, 1)
        layout.add_widget(b)

        layout.size_hint = (1, 1)
        root.add_widget(layout)

        #Clock.schedule_interval(self.update, 100)
        Clock.schedule_interval(lambda dt: b.reload(), 0.01) #speed that image updates
        return root

    def clear_canvas(self, obj):
        self.painter.canvas.clear()

    #def update(self):
        #self.name

SimpleKivy4().run()
