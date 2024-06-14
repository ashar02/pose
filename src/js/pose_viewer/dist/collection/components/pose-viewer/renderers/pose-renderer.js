export class PoseRenderer {
  constructor(viewer) {
    this.viewer = viewer;
  }
  x(v) {
    const n = v * (this.viewer.elWidth - 2 * this.viewer.elPadding.width);
    return n / this.viewer.pose.header.width + this.viewer.elPadding.width;
  }
  y(v) {
    const n = v * (this.viewer.elHeight - 2 * this.viewer.elPadding.height);
    return n / this.viewer.pose.header.height + this.viewer.elPadding.height;
  }
  isJointValid(joint) {
    return joint.C > 0;
  }
  renderJoints(joints, colors) {
    return joints
      .filter(this.isJointValid.bind(this))
      .map((joint, i) => {
      return this.renderJoint(i, joint, colors[i % colors.length]);
    });
  }
  renderLimbs(limbs, joints, colors) {
    return limbs.map(({ from, to }) => {
      const a = joints[from];
      const b = joints[to];
      if (!this.isJointValid(a) || !this.isJointValid(b)) {
        return "";
      }
      const c1 = colors[from % colors.length];
      const c2 = colors[to % colors.length];
      const color = {
        R: (c1.R + c2.R) / 2,
        G: (c1.G + c2.G) / 2,
        B: (c1.B + c2.B) / 2,
      };
      return this.renderLimb(a, b, color);
    });
  }
  renderFrame(frame) {
    return frame.people.map(person => this.viewer.pose.header.components.map(component => {
      const joints = person[component.name];
      return [
        this.renderJoints(joints, component.colors),
        this.renderLimbs(component.limbs, joints, component.colors),
      ];
    }));
  }
}
//# sourceMappingURL=pose-renderer.js.map
