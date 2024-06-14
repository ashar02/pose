// @ts-ignore
import { h, Host } from "@stencil/core";
import { Pose } from "pose-format";
import { SVGPoseRenderer } from "./renderers/svg.pose-renderer";
import { CanvasPoseRenderer } from "./renderers/canvas.pose-renderer";
export class PoseViewer {
  constructor() {
    // @Event() ratechange$: EventEmitter<void>;
    // @Event() seeked$: EventEmitter<void>;
    // @Event() seeking$: EventEmitter<void>;
    // @Event() timeupdate$: EventEmitter<void>;
    this.hasRendered = false;
    this.src = undefined;
    this.svg = false;
    this.width = null;
    this.height = null;
    this.aspectRatio = null;
    this.padding = null;
    this.thickness = null;
    this.background = null;
    this.loop = false;
    this.autoplay = true;
    this.playbackRate = 1;
    this.currentTime = NaN;
    this.duration = NaN;
    this.ended = false;
    this.paused = true;
    this.readyState = 0;
    this.error = undefined;
  }
  componentWillLoad() {
    this.renderer = this.svg ? new SVGPoseRenderer(this) : new CanvasPoseRenderer(this);
    return this.srcChange();
  }
  componentDidLoad() {
    this.resizeObserver = new ResizeObserver(this.setDimensions.bind(this));
    this.resizeObserver.observe(this.element);
  }
  async getRemotePose() {
    // Abort previous request
    if (this.fetchAbortController) {
      this.fetchAbortController.abort();
    }
    this.fetchAbortController = new AbortController();
    this.pose = await Pose.fromRemote(this.src, this.fetchAbortController);
  }
  initPose() {
    this.setDimensions();
    // Loaded done events
    this.loadedmetadata$.emit();
    this.loadeddata$.emit();
    this.canplaythrough$.emit();
    this.duration = (this.pose.body.frames.length - 1) / this.pose.body.fps;
    this.currentTime = 0;
    if (this.autoplay) {
      this.play();
    }
  }
  async srcChange() {
    // Can occur from both an attribute change AND componentWillLoad event
    if (this.src === this.lastSrc) {
      return;
    }
    this.lastSrc = this.src;
    // Clear previous pose
    this.clearInterval();
    this.setDimensions();
    delete this.pose;
    this.currentTime = NaN;
    this.duration = NaN;
    this.hasRendered = false;
    if (!this.src) {
      return;
    }
    // Load new pose
    this.ended = false;
    this.loadstart$.emit();
    this.error = null;
    try {
      await this.getRemotePose();
      this.initPose();
      this.error = null;
    }
    catch (e) {
      console.error('PoseViewer error', e);
      this.error = e;
    }
  }
  setDimensions() {
    this.elPadding = { width: 0, height: 0 };
    if (!this.pose) {
      this.elWidth = 0;
      this.elHeight = 0;
      return;
    }
    // When nothing is marked, use pose dimensions
    if (!this.width && !this.height) {
      this.elWidth = this.pose.header.width;
      this.elHeight = this.pose.header.height;
      return;
    }
    const rect = this.element.getBoundingClientRect();
    const parseSize = (size, by) => size.endsWith("px") ? Number(size.slice(0, -2)) : (size.endsWith("%") ? by * size.slice(0, -1) / 100 : Number(size));
    // When both are marked,
    if (this.width && this.height) {
      this.elWidth = parseSize(this.width, rect.width);
      this.elHeight = parseSize(this.height, rect.height);
    }
    else if (this.width) {
      this.elWidth = parseSize(this.width, rect.width);
      this.elHeight = this.aspectRatio ? this.elWidth * this.aspectRatio :
        (this.pose.header.height / this.pose.header.width) * this.elWidth;
    }
    else if (this.height) {
      this.elHeight = parseSize(this.height, rect.height);
      this.elWidth = this.aspectRatio ? this.elHeight / this.aspectRatio :
        (this.pose.header.width / this.pose.header.height) * this.elHeight;
    }
    // General padding
    if (this.padding) {
      this.elPadding.width += parseSize(this.padding, this.elWidth);
      this.elPadding.height += parseSize(this.padding, this.elHeight);
    }
    // Aspect ratio padding
    const ratioWidth = this.elWidth - this.elPadding.width * 2;
    const ratioHeight = this.elHeight - this.elPadding.height * 2;
    const elAR = ratioWidth / ratioHeight;
    const poseAR = this.pose.header.width / this.pose.header.height;
    if (poseAR > elAR) {
      this.elPadding.height += (poseAR - elAR) * ratioHeight / 2;
    }
    else {
      this.elPadding.width += (1 / poseAR - 1 / elAR) * ratioWidth / 2;
    }
  }
  async syncMedia(media) {
    this.media = media;
    this.media.addEventListener('pause', this.pause.bind(this));
    this.media.addEventListener('play', this.play.bind(this));
    const syncTime = () => this.currentTime = this.frameTime(this.media.currentTime);
    this.media.addEventListener('seek', syncTime);
    this.media.addEventListener('timeupdate', syncTime); // To always keep synced
    // Others
    const updateRate = () => this.playbackRate = this.media.playbackRate;
    this.media.addEventListener('ratechange', updateRate);
    updateRate();
    // Start the pose according to the video
    this.clearInterval();
    if (this.media.paused) {
      this.pause();
    }
    else {
      this.play();
    }
  }
  async getPose() {
    return this.pose;
  }
  async nextFrame() {
    const newTime = this.currentTime + 1 / this.pose.body.fps;
    if (newTime > this.duration) {
      if (this.loop) {
        this.currentTime = newTime % this.duration;
      }
      else {
        this.ended$.emit();
        this.ended = true;
      }
    }
    else {
      this.currentTime = newTime;
    }
  }
  frameTime(time) {
    if (!this.pose) {
      return 0;
    }
    return Math.floor(time * this.pose.body.fps) / this.pose.body.fps;
  }
  async play() {
    if (!this.paused) {
      this.clearInterval();
    }
    this.paused = false;
    this.play$.emit();
    // Reset clip if exceeded duration
    if (this.currentTime > this.duration) {
      this.currentTime = 0;
    }
    const intervalTime = 1000 / (this.pose.body.fps * this.playbackRate);
    if (this.media) {
      this.loopInterval = setInterval(() => this.currentTime = this.frameTime(this.media.currentTime), intervalTime);
    }
    else {
      // Add the time passed in an interval.
      let lastTime = Date.now() / 1000;
      this.loopInterval = setInterval(() => {
        const now = Date.now() / 1000;
        this.currentTime += (now - lastTime) * this.playbackRate;
        lastTime = now;
        if (this.currentTime > this.duration) {
          if (this.loop) {
            this.currentTime = this.currentTime % this.duration;
          }
          else {
            this.ended$.emit();
            this.ended = true;
            this.clearInterval();
          }
        }
      }, intervalTime);
    }
  }
  async pause() {
    this.paused = true;
    this.pause$.emit();
    this.clearInterval();
  }
  clearInterval() {
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
    }
  }
  disconnectedCallback() {
    this.clearInterval();
  }
  render() {
    if (this.error) {
      return this.error.name !== "AbortError" ? this.error.message : "";
    }
    if (!this.pose || isNaN(this.currentTime) || !this.renderer) {
      return "";
    }
    const currentTime = this.currentTime > this.duration ? this.duration : this.currentTime;
    const frameId = Math.floor(currentTime * this.pose.body.fps);
    const frame = this.pose.body.frames[frameId];
    const render = this.renderer.render(frame);
    if (!this.hasRendered) {
      requestAnimationFrame(() => {
        this.hasRendered = true;
        this.firstRender$.emit();
      });
    }
    requestAnimationFrame(() => this.render$.emit());
    return (h(Host, null, render));
  }
  static get is() { return "pose-viewer"; }
  static get encapsulation() { return "shadow"; }
  static get originalStyleUrls() {
    return {
      "$": ["pose-viewer.css"]
    };
  }
  static get styleUrls() {
    return {
      "$": ["pose-viewer.css"]
    };
  }
  static get properties() {
    return {
      "src": {
        "type": "string",
        "mutable": false,
        "complexType": {
          "original": "string",
          "resolved": "string",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "src",
        "reflect": false
      },
      "svg": {
        "type": "boolean",
        "mutable": false,
        "complexType": {
          "original": "boolean",
          "resolved": "boolean",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "svg",
        "reflect": false,
        "defaultValue": "false"
      },
      "width": {
        "type": "string",
        "mutable": false,
        "complexType": {
          "original": "string",
          "resolved": "string",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "width",
        "reflect": false,
        "defaultValue": "null"
      },
      "height": {
        "type": "string",
        "mutable": false,
        "complexType": {
          "original": "string",
          "resolved": "string",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "height",
        "reflect": false,
        "defaultValue": "null"
      },
      "aspectRatio": {
        "type": "number",
        "mutable": false,
        "complexType": {
          "original": "number",
          "resolved": "number",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "aspect-ratio",
        "reflect": false,
        "defaultValue": "null"
      },
      "padding": {
        "type": "string",
        "mutable": false,
        "complexType": {
          "original": "string",
          "resolved": "string",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "padding",
        "reflect": false,
        "defaultValue": "null"
      },
      "thickness": {
        "type": "number",
        "mutable": false,
        "complexType": {
          "original": "number",
          "resolved": "number",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "thickness",
        "reflect": false,
        "defaultValue": "null"
      },
      "background": {
        "type": "string",
        "mutable": false,
        "complexType": {
          "original": "string",
          "resolved": "string",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "background",
        "reflect": false,
        "defaultValue": "null"
      },
      "loop": {
        "type": "boolean",
        "mutable": true,
        "complexType": {
          "original": "boolean",
          "resolved": "boolean",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "loop",
        "reflect": false,
        "defaultValue": "false"
      },
      "autoplay": {
        "type": "boolean",
        "mutable": false,
        "complexType": {
          "original": "boolean",
          "resolved": "boolean",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "autoplay",
        "reflect": false,
        "defaultValue": "true"
      },
      "playbackRate": {
        "type": "number",
        "mutable": true,
        "complexType": {
          "original": "number",
          "resolved": "number",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "playback-rate",
        "reflect": false,
        "defaultValue": "1"
      },
      "currentTime": {
        "type": "number",
        "mutable": true,
        "complexType": {
          "original": "number",
          "resolved": "number",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "current-time",
        "reflect": false,
        "defaultValue": "NaN"
      },
      "duration": {
        "type": "number",
        "mutable": true,
        "complexType": {
          "original": "number",
          "resolved": "number",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "duration",
        "reflect": false,
        "defaultValue": "NaN"
      },
      "ended": {
        "type": "boolean",
        "mutable": true,
        "complexType": {
          "original": "boolean",
          "resolved": "boolean",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "ended",
        "reflect": false,
        "defaultValue": "false"
      },
      "paused": {
        "type": "boolean",
        "mutable": true,
        "complexType": {
          "original": "boolean",
          "resolved": "boolean",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "paused",
        "reflect": false,
        "defaultValue": "true"
      },
      "readyState": {
        "type": "number",
        "mutable": true,
        "complexType": {
          "original": "number",
          "resolved": "number",
          "references": {}
        },
        "required": false,
        "optional": false,
        "docs": {
          "tags": [],
          "text": ""
        },
        "attribute": "ready-state",
        "reflect": false,
        "defaultValue": "0"
      }
    };
  }
  static get states() {
    return {
      "error": {}
    };
  }
  static get events() {
    return [{
        "method": "canplaythrough$",
        "name": "canplaythrough$",
        "bubbles": true,
        "cancelable": true,
        "composed": true,
        "docs": {
          "tags": [],
          "text": ""
        },
        "complexType": {
          "original": "void",
          "resolved": "void",
          "references": {}
        }
      }, {
        "method": "ended$",
        "name": "ended$",
        "bubbles": true,
        "cancelable": true,
        "composed": true,
        "docs": {
          "tags": [],
          "text": ""
        },
        "complexType": {
          "original": "void",
          "resolved": "void",
          "references": {}
        }
      }, {
        "method": "loadeddata$",
        "name": "loadeddata$",
        "bubbles": true,
        "cancelable": true,
        "composed": true,
        "docs": {
          "tags": [],
          "text": ""
        },
        "complexType": {
          "original": "void",
          "resolved": "void",
          "references": {}
        }
      }, {
        "method": "loadedmetadata$",
        "name": "loadedmetadata$",
        "bubbles": true,
        "cancelable": true,
        "composed": true,
        "docs": {
          "tags": [],
          "text": ""
        },
        "complexType": {
          "original": "void",
          "resolved": "void",
          "references": {}
        }
      }, {
        "method": "loadstart$",
        "name": "loadstart$",
        "bubbles": true,
        "cancelable": true,
        "composed": true,
        "docs": {
          "tags": [],
          "text": ""
        },
        "complexType": {
          "original": "void",
          "resolved": "void",
          "references": {}
        }
      }, {
        "method": "pause$",
        "name": "pause$",
        "bubbles": true,
        "cancelable": true,
        "composed": true,
        "docs": {
          "tags": [],
          "text": ""
        },
        "complexType": {
          "original": "void",
          "resolved": "void",
          "references": {}
        }
      }, {
        "method": "play$",
        "name": "play$",
        "bubbles": true,
        "cancelable": true,
        "composed": true,
        "docs": {
          "tags": [],
          "text": ""
        },
        "complexType": {
          "original": "void",
          "resolved": "void",
          "references": {}
        }
      }, {
        "method": "firstRender$",
        "name": "firstRender$",
        "bubbles": true,
        "cancelable": true,
        "composed": true,
        "docs": {
          "tags": [],
          "text": ""
        },
        "complexType": {
          "original": "void",
          "resolved": "void",
          "references": {}
        }
      }, {
        "method": "render$",
        "name": "render$",
        "bubbles": true,
        "cancelable": true,
        "composed": true,
        "docs": {
          "tags": [],
          "text": ""
        },
        "complexType": {
          "original": "void",
          "resolved": "void",
          "references": {}
        }
      }];
  }
  static get methods() {
    return {
      "syncMedia": {
        "complexType": {
          "signature": "(media: HTMLMediaElement) => Promise<void>",
          "parameters": [{
              "tags": [],
              "text": ""
            }],
          "references": {
            "Promise": {
              "location": "global",
              "id": "global::Promise"
            },
            "HTMLMediaElement": {
              "location": "global",
              "id": "global::HTMLMediaElement"
            }
          },
          "return": "Promise<void>"
        },
        "docs": {
          "text": "",
          "tags": []
        }
      },
      "getPose": {
        "complexType": {
          "signature": "() => Promise<PoseModel>",
          "parameters": [],
          "references": {
            "Promise": {
              "location": "global",
              "id": "global::Promise"
            },
            "PoseModel": {
              "location": "import",
              "path": "pose-format/dist/types",
              "id": ""
            }
          },
          "return": "Promise<PoseModel>"
        },
        "docs": {
          "text": "",
          "tags": []
        }
      },
      "nextFrame": {
        "complexType": {
          "signature": "() => Promise<void>",
          "parameters": [],
          "references": {
            "Promise": {
              "location": "global",
              "id": "global::Promise"
            }
          },
          "return": "Promise<void>"
        },
        "docs": {
          "text": "",
          "tags": []
        }
      },
      "play": {
        "complexType": {
          "signature": "() => Promise<void>",
          "parameters": [],
          "references": {
            "Promise": {
              "location": "global",
              "id": "global::Promise"
            }
          },
          "return": "Promise<void>"
        },
        "docs": {
          "text": "",
          "tags": []
        }
      },
      "pause": {
        "complexType": {
          "signature": "() => Promise<void>",
          "parameters": [],
          "references": {
            "Promise": {
              "location": "global",
              "id": "global::Promise"
            }
          },
          "return": "Promise<void>"
        },
        "docs": {
          "text": "",
          "tags": []
        }
      }
    };
  }
  static get elementRef() { return "element"; }
  static get watchers() {
    return [{
        "propName": "src",
        "methodName": "srcChange"
      }];
  }
}
//# sourceMappingURL=pose-viewer.js.map
