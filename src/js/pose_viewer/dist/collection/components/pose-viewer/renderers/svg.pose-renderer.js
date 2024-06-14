import { PoseRenderer } from "./pose-renderer";
import { h } from "@stencil/core";
export class SVGPoseRenderer extends PoseRenderer {
  renderJoint(i, joint, color) {
    const { R, G, B } = color;
    return (h("circle", { cx: joint.X, cy: joint.Y, r: 4, class: "joint draggable", style: ({
        fill: `rgb(${R}, ${G}, ${B})`,
        opacity: String(joint.C)
      }), "data-id": i }));
  }
  renderLimb(from, to, color) {
    const { R, G, B } = color;
    return (h("line", { x1: from.X, y1: from.Y, x2: to.X, y2: to.Y, style: {
        stroke: `rgb(${R}, ${G}, ${B})`,
        opacity: String((from.C + to.C) / 2)
      } }));
  }
  render(frame) {
    const viewBox = `0 0 ${this.viewer.pose.header.width} ${this.viewer.pose.header.height}`;
    return (h("svg", { xmlns: "http://www.w3.org/2000/svg", viewBox: viewBox, width: this.viewer.elWidth, height: this.viewer.elHeight }, h("g", null, this.renderFrame(frame))));
  }
}
//# sourceMappingURL=svg.pose-renderer.js.map
