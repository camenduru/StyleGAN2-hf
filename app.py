#!/usr/bin/env python

from __future__ import annotations

import argparse
import json

import gradio as gr
import numpy as np

from model import Model

TITLE = '# StyleGAN2'
DESCRIPTION = '''This is an unofficial demo for [https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan3).

Expected execution time on Hugging Face Spaces: 4s
'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.stylegan2" />'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def update_class_index(name: str) -> dict:
    if name == 'CIFAR-10':
        return gr.Slider.update(maximum=9, visible=True)
    else:
        return gr.Slider.update(visible=False)


def get_sample_image_url(name: str) -> str:
    sample_image_dir = 'https://huggingface.co/spaces/hysts/StyleGAN2/resolve/main/samples'
    return f'{sample_image_dir}/{name}.jpg'


def get_sample_image_markdown(name: str) -> str:
    url = get_sample_image_url(name)
    if name == 'cifar10':
        size = 32
        class_index = '0-9'
        seed = '0-9'
    else:
        class_index = 'N/A'
        seed = '0-99'
        if name == 'afhq-cat':
            size = 512
        elif name == 'afhq-dog':
            size = 512
        elif name == 'afhq-wild':
            size = 512
        elif name == 'afhqv2':
            size = 512
        elif name == 'brecahad':
            size = 256
        elif name == 'celebahq':
            size = 1024
        elif name == 'ffhq':
            size = 1024
        elif name == 'ffhq-u':
            size = 1024
        elif name == 'lsun-dog':
            size = 256
        elif name == 'metfaces':
            size = 1024
        elif name == 'metfaces-u':
            size = 1024
        else:
            raise ValueError

    return f'''
    - size: {size}x{size}
    - class_index: {class_index}
    - seed: {seed}
    - truncation: 0.7
    ![sample images]({url})'''


def load_class_names(name: str) -> list[str]:
    with open(f'labels/{name}_classes.json') as f:
        names = json.load(f)
    return names


def get_class_name_df(name: str) -> list:
    names = load_class_names(name)
    return list(map(list, enumerate(names)))  # type: ignore


CIFAR10_NAMES = load_class_names('cifar10')


def update_class_name(model_name: str, index: int) -> dict:
    if model_name == 'CIFAR-10':
        value = CIFAR10_NAMES[index]
        return gr.Textbox.update(value=value, visible=True)
    else:
        return gr.Textbox.update(visible=False)


def main():
    args = parse_args()
    model = Model(args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        with gr.Tabs():
            with gr.TabItem('App'):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            model_name = gr.Dropdown(list(
                                model.MODEL_NAME_DICT.keys()),
                                                     value='FFHQ-1024',
                                                     label='Model')
                            seed = gr.Slider(0,
                                             np.iinfo(np.uint32).max,
                                             step=1,
                                             value=0,
                                             label='Seed')
                            psi = gr.Slider(0,
                                            2,
                                            step=0.05,
                                            value=0.7,
                                            label='Truncation psi')
                            class_index = gr.Slider(0,
                                                    9,
                                                    step=1,
                                                    value=0,
                                                    label='Class Index',
                                                    visible=False)
                            class_name = gr.Textbox(
                                value=CIFAR10_NAMES[class_index.value],
                                label='Class Label',
                                interactive=False,
                                visible=False)
                            run_button = gr.Button('Run')
                    with gr.Column():
                        result = gr.Image(label='Result', elem_id='result')

            with gr.TabItem('Sample Images'):
                with gr.Row():
                    model_name2 = gr.Dropdown([
                        'afhq-cat',
                        'afhq-dog',
                        'afhq-wild',
                        'afhqv2',
                        'brecahad',
                        'celebahq',
                        'cifar10',
                        'ffhq',
                        'ffhq-u',
                        'lsun-dog',
                        'metfaces',
                        'metfaces-u',
                    ],
                                              value='afhq-cat',
                                              label='Model')
                with gr.Row():
                    text = get_sample_image_markdown(model_name2.value)
                    sample_images = gr.Markdown(text)

        gr.Markdown(FOOTER)

        model_name.change(fn=model.set_model, inputs=model_name, outputs=None)
        model_name.change(fn=update_class_index,
                          inputs=model_name,
                          outputs=class_index)
        model_name.change(fn=update_class_name,
                          inputs=[
                              model_name,
                              class_index,
                          ],
                          outputs=class_name)
        class_index.change(fn=update_class_name,
                           inputs=[
                               model_name,
                               class_index,
                           ],
                           outputs=class_name)
        run_button.click(fn=model.set_model_and_generate_image,
                         inputs=[
                             model_name,
                             seed,
                             psi,
                             class_index,
                         ],
                         outputs=result)
        model_name2.change(fn=get_sample_image_markdown,
                           inputs=model_name2,
                           outputs=sample_images)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
