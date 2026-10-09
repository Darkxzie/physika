Two-dimensional shallow water with Runge–Kutta time stepping
============================================================

A raised patch of water pushes on its surroundings. Water flows away from the
patch, the surface falls, and gravity waves carry the disturbance across the
surface. This tutorial models that motion on a periodic square, starting from
a smooth bump with no initial horizontal flow.

The program uses a first-order finite-volume method in space and a three-stage,
third-order strong-stability-preserving Runge–Kutta method (SSPRK3) in time.
We derive the update from conservation of water and horizontal momentum, then
connect each part to ``tutorials/shallow_water_rk3.phyk``. The complete program
is at the end. For the language basics and installation, see
:doc:`intro_to_physika` and :doc:`/install`.

What “shallow water” means
--------------------------

The simulated water has a depth, but the computational grid covers only the two
horizontal directions. Each grid cell represents an entire water column.
The velocities are averages through that column; the model does not resolve a
vertical velocity profile or a three-dimensional flow field. A surface plot
with height on its vertical axis would still show a two-dimensional model.

The shallow-water approximation applies when the vertical length scale is
small compared with the horizontal scale of the motion. Vertical acceleration
is neglected in the pressure balance, giving hydrostatic pressure. This
example assumes a flat, stationary bed and omits friction, wind, rotation,
bathymetry, and external sources of water or momentum.

Let :math:`x` and :math:`y` be horizontal coordinates and :math:`t` be time.
The total water depth is :math:`h(x,y,t)`, the constant reference depth is
:math:`H`, and :math:`\eta(x,y,t)` is the surface displacement from that
reference level:

.. math::

   h = H + \eta, \qquad \eta = h-H.

Thus ``h`` is not just the bump height. The bump is the smaller displacement
:math:`\eta`. The horizontal depth-averaged velocities are :math:`u(x,y,t)`
in the positive :math:`x` direction and :math:`v(x,y,t)` in the positive
:math:`y` direction. The positive parameter :math:`g` is gravity in the units
used by the simulation.

Units and scales
~~~~~~~~~~~~~~~~

All numbers in the program are dimensionless. One consistent conversion to
physical units uses a horizontal length scale :math:`L_*`, a depth scale
:math:`D_*`, and physical gravitational acceleration :math:`g_*`. The star
on these constants marks a dimensional scale. Define the velocity and time
scales :math:`U_*` and :math:`T_*` by

.. math::

   U_* = \sqrt{g_*D_*}, \qquad T_* = \frac{L_*}{U_*}.

With the subscript “phys” denoting dimensional quantities, the conversion is

.. math::

   x_{\mathrm{phys}}=L_*x, \quad y_{\mathrm{phys}}=L_*y,
   \quad t_{\mathrm{phys}}=T_*t, \quad
   h_{\mathrm{phys}}=D_*h, \quad
   u_{\mathrm{phys}}=U_*u, \quad v_{\mathrm{phys}}=U_*v,
   \qquad g=\frac{g_*D_*}{U_*^2}=1.

The reference depth and surface displacement use the same depth scale as
:math:`h`. Setting the dimensionless reference depth to one does not say the
physical water is as deep as the domain is wide: vertical and horizontal
lengths use different scales. A shallow physical interpretation requires
:math:`D_*/L_*` to be small, and the horizontal scale of the bump must also be
large compared with the depth. The code does not choose those physical scales.

From moving water to conservation laws
--------------------------------------

Water entering a cell increases its depth; water leaving decreases it. Momentum
is carried with the moving water, and pressure also transfers momentum between
neighbouring columns. We therefore evolve depth and the two momentum
components together rather than updating velocities independently.

The *primitive* fields are depth and velocities. The *conserved* state
:math:`\mathbf q` and the recovery of its primitive fields are

.. math::

   \mathbf q = \begin{pmatrix}h\\hu\\hv\end{pmatrix},
   \qquad u=\frac{hu}{h}, \qquad v=\frac{hv}{h}, \qquad h>0.

Here :math:`hu` and :math:`hv` mean products, not extra velocity variables.
For constant density they are horizontal momenta per unit horizontal area
with density divided out. Their integrals, and the integral of depth, are
conserved on this periodic domain.

Let :math:`\mathbf F` be the vector of physical fluxes in the positive
:math:`x` direction and :math:`\mathbf G` the corresponding vector in the
positive :math:`y` direction. A flux measures transport across a face per unit
face length and per unit time. The equations are

.. math::

   \frac{\partial\mathbf q}{\partial t}
   +\frac{\partial\mathbf F(\mathbf q)}{\partial x}
   +\frac{\partial\mathbf G(\mathbf q)}{\partial y}=\mathbf 0,

   \mathbf F(\mathbf q)=
   \begin{pmatrix}hu\\hu^2+\tfrac12gh^2\\huv\end{pmatrix},
   \qquad
   \mathbf G(\mathbf q)=
   \begin{pmatrix}hv\\huv\\hv^2+\tfrac12gh^2\end{pmatrix}.

The symbol :math:`\partial` denotes a partial derivative, varying one
coordinate while holding the others fixed; :math:`\mathbf 0` is a vector of
three zeros. The first row conserves water, the second conserves
:math:`x` momentum, and the third conserves :math:`y` momentum.
The hydrostatic contribution used in the code is

.. math::

   p=\frac12gh^2.

Here :math:`p` is depth-integrated pressure divided by density, named
``pressure`` in the program. It is not the pressure at a single vertical
position. The remaining momentum-flux terms describe transport by the flow.
This nonlinear system is the PDE (partial differential equation) we discretize.

From the PDE to cell averages
-----------------------------

Divide the unit square into :math:`N_x` cells along :math:`x` and :math:`N_y`
cells along :math:`y`. Their widths and centre coordinates are

.. math::

   \Delta x=\frac1{N_x}, \qquad \Delta y=\frac1{N_y}, \qquad
   x_i=(i+\tfrac12)\Delta x, \qquad y_j=(j+\tfrac12)\Delta y,
   \qquad
   i=0,\ldots,N_x-1, \quad j=0,\ldots,N_y-1.

The integer indices :math:`i` and :math:`j` label cells, not time steps.
Neither endpoint is a cell centre. Let :math:`C_{ij}` denote the rectangular
cell centred at :math:`(x_i,y_j)`, whose area is
:math:`\Delta x\Delta y`. Its exact average state is

.. math::

   \overline{\mathbf q}_{ij}(t)
   =\frac1{\Delta x\Delta y}\int_{C_{ij}}\mathbf q(x,y,t)\,dx\,dy.

The bar denotes an average over the cell; the integral sums the field over
its area. Integrating the PDE over the cell turns spatial derivatives into
boundary fluxes:

.. math::

   \frac{d\overline{\mathbf q}_{ij}}{dt}
   =-\frac{\overline{\mathbf F}_{i+1/2,j}
                  -\overline{\mathbf F}_{i-1/2,j}}{\Delta x}
    -\frac{\overline{\mathbf G}_{i,j+1/2}
                  -\overline{\mathbf G}_{i,j-1/2}}{\Delta y}.

Here :math:`d/dt` is the time derivative of a cell average. The barred fluxes
are exact averages of the physical flux along each cell face. Half-integer
indices label faces: :math:`i+1/2` is the right face, :math:`i-1/2` the left,
:math:`j+1/2` the top, and :math:`j-1/2` the bottom. The minus signs express
“incoming minus outgoing”: excess outward flux reduces the stored quantity.

The numerical array :math:`\mathbf q_{ij}` approximates
:math:`\overline{\mathbf q}_{ij}`. Below we omit the bar for numerical states
and replace the unknown exact face fluxes with computable numerical fluxes.
Within each cell the method uses a constant state; it does not reconstruct
slopes. That choice makes the spatial method first order.

Array axes and periodic neighbours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete ``q`` array has type ``ℝ[3, m, n]``: three conserved components,
then the two spatial dimensions. In ``q[k, i, j]``, the component index ``k``
is 0 for depth, 1 for x momentum, or 2 for y momentum. ``ℝ`` denotes real numbers; ``ℕ`` denotes
natural-number counts and indices. The shape parameters ``m`` and ``n`` stand
for the two grid sizes, instantiated as ``Nx`` and ``Ny``. This code-shape
``n`` is unrelated to the time index used below.

After extracting ``q[0]``, ``q[1]``, or ``q[2]``, each component is a
``ℝ[m, n]`` array. On these component arrays **axis 0 is x and axis 1 is y**.
These are not the spatial axis numbers of the unsliced three-dimensional
state array. If plotting with software that treats the first index as a
vertical image row, account for this convention, for example by transposing
the component array when placing x horizontally.

``roll(w, -1, 0)`` places the next x-cell value at each current cell;
``roll(w, 1, 0)`` supplies the previous x-cell value. Axis 1 does the same in
y. Here ``w`` is any one component field. Algebraically,

.. math::

   [\operatorname{roll}(w,-1,0)]_{ij}=w_{i+1,j}, \qquad
   [\operatorname{roll}(w,-1,1)]_{ij}=w_{i,j+1}.

The operator :math:`\operatorname{roll}` shifts an array with wrap-around.
All spatial indices wrap modulo their respective cell counts: the neighbour
after the last x-cell is the first x-cell, and similarly in y. Opposite edges
are joined, so a wave leaving one edge re-enters through the opposite edge.
These are periodic boundaries, not reflecting walls or open boundaries.

Rusanov fluxes at cell faces
----------------------------

Two adjacent cells usually predict different physical fluxes. We need one
shared flux so that whatever leaves one cell enters its neighbour. The local
Rusanov flux averages the two physical fluxes and adds a damping term
proportional to their state jump. The damping is numerical diffusion, not a
physical viscosity specified for the water.

Gravity waves move relative to the fluid with speed :math:`c`. The nonnegative
cell speed bounds :math:`s^x` and :math:`s^y` combine advection and waves:

.. math::

   c_{ij}=\sqrt{g h_{ij}}, \qquad
   s^x_{ij}=|u_{ij}|+c_{ij}, \qquad
   s^y_{ij}=|v_{ij}|+c_{ij}.

The absolute-value bars give speed regardless of the sign of velocity.
Superscripts x and y label directions, not powers. The wave-speed bounds at
the right and top faces are :math:`a^x` and :math:`a^y`:

.. math::

   a^x_{i+1/2,j}=\max(s^x_{ij},s^x_{i+1,j}), \qquad
   a^y_{i,j+1/2}=\max(s^y_{ij},s^y_{i,j+1}).

The maximum selects the larger speed from the two cells sharing a face.
The ``face_speed`` function uses the elementwise identity

.. math::

   \max(a,b)=\tfrac12(a+b+|a-b|),

where :math:`a` and :math:`b` are any two real values. In the function, ``a``
is the cell-speed array, ``neighbor`` is its forward neighbour array, and
``axis`` selects the spatial direction.

.. code-block:: text

   def face_speed(a: ℝ[m, n], axis: ℕ): ℝ[m, n]:
       neighbor: ℝ[m, n] = roll(a, -1, axis)
       return 0.5 * (a + neighbor + abs(a - neighbor))

Use a hat to distinguish a numerical face flux from a physical flux evaluated
inside a cell. The vector Rusanov fluxes are

.. math::

   \widehat{\mathbf F}_{i+1/2,j}
   =\tfrac12\left[\mathbf F(\mathbf q_{ij})+
                          \mathbf F(\mathbf q_{i+1,j})\right]
    -\tfrac12a^x_{i+1/2,j}(\mathbf q_{i+1,j}-\mathbf q_{ij}),

   \widehat{\mathbf G}_{i,j+1/2}
   =\tfrac12\left[\mathbf G(\mathbf q_{ij})+
                          \mathbf G(\mathbf q_{i,j+1})\right]
    -\tfrac12a^y_{i,j+1/2}(\mathbf q_{i,j+1}-\mathbf q_{ij}).

The left and right states at an x-face are simply its adjacent cell states;
the lower and upper states play the same role at a y-face. The same face
speed multiplies the jump in each of the three conserved components.

From fluxes to the right-hand side
----------------------------------

Replacing the exact face averages in the cell balance defines the spatial
operator :math:`\mathcal L`, which maps the entire grid state to its rate of
change:

.. math::

   \frac{d\mathbf q_{ij}}{dt}=\mathcal L(\mathbf q)_{ij},
   \qquad
   \mathcal L(\mathbf q)_{ij}
   =-\frac{\widehat{\mathbf F}_{i+1/2,j}
                   -\widehat{\mathbf F}_{i-1/2,j}}{\Delta x}
    -\frac{\widehat{\mathbf G}_{i,j+1/2}
                   -\widehat{\mathbf G}_{i,j-1/2}}{\Delta y}.

This is an ODE (ordinary differential equation) for the array values: space
has been discretized, but time is still continuous. The function
``divergence`` computes one component of this negative flux divergence.
Its ``w`` argument is the conserved component; ``f`` and ``flux_y`` are that
component's physical x- and y-fluxes. ``ax`` and ``ay`` are the face-speed
arrays, and ``dx`` and ``dy`` are the cell widths. The arrays ``F`` and
``G`` store the numerical fluxes on each cell's right and top faces.
Shifting those fluxes backwards supplies the left and bottom faces.

.. code-block:: text

   def divergence(w: ℝ[m, n], f: ℝ[m, n], flux_y: ℝ[m, n],
                  ax: ℝ[m, n], ay: ℝ[m, n], dx: ℝ, dy: ℝ): ℝ[m, n]:
       F: ℝ[m, n] = 0.5 * (f + roll(f, -1, 0))
       G: ℝ[m, n] = 0.5 * (flux_y + roll(flux_y, -1, 1))
       F = F - 0.5 * ax * (roll(w, -1, 0) - w)
       G = G - 0.5 * ay * (roll(w, -1, 1) - w)
       return -(F - roll(F, 1, 0)) / dx - (G - roll(G, 1, 1)) / dy

The function ``rhs`` means “right-hand side” and implements
:math:`\mathcal L`. It recovers velocities from the input state, computes wave
speeds and pressure, then calls ``divergence`` for depth and both momenta.
The returned ``dh``, ``dhu``, and ``dhv`` are time derivatives, not yet changes
over a time step.

.. code-block:: text

   def rhs(q: ℝ[3, m, n], g: ℝ, dx: ℝ, dy: ℝ): ℝ[3, m, n]:
       h: ℝ[m, n] = q[0]
       hu: ℝ[m, n] = q[1]
       hv: ℝ[m, n] = q[2]
       u: ℝ[m, n] = hu / h
       v: ℝ[m, n] = hv / h
       c: ℝ[m, n] = sqrt(g * h)
       ax: ℝ[m, n] = face_speed(abs(u) + c, 0)
       ay: ℝ[m, n] = face_speed(abs(v) + c, 1)
       pressure: ℝ[m, n] = 0.5 * g * h * h
       dh: ℝ[m, n] = divergence(h, hu, hv, ax, ay, dx, dy)
       dhu: ℝ[m, n] = divergence(hu, hu * u + pressure, hv * u, ax, ay, dx, dy)
       dhv: ℝ[m, n] = divergence(hv, hu * v, hv * v + pressure, ax, ay, dx, dy)
       return [dh, dhu, dhv]

On a periodic grid, summing a flux difference over all cells cancels every
face contribution with its neighbour's. This explains discrete conservation
of total depth and momentum in exact arithmetic. Floating-point roundoff can
leave a small residual.

Advancing time with SSPRK3
--------------------------

A forward Euler step adds the current rate of change multiplied by the time
step. SSPRK3 combines three such evaluations, allowing each new state to
change the velocities, pressure, and wave speeds used by the next evaluation.
Keeping all three conserved components together avoids using stale velocities.

Let :math:`n` be the completed time-step index, :math:`\Delta t` the fixed time
step, and :math:`t^n=n\Delta t` the time after that many steps, starting at
zero. The full grid state there is :math:`\mathbf q^n`. Define the Euler map
:math:`E` and the internal RK states by

.. math::

   E(\mathbf q)=\mathbf q+\Delta t\,\mathcal L(\mathbf q),

   \begin{aligned}
   \mathbf q^{(1)} &= E(\mathbf q^n),\\
   \mathbf q^{(2)} &= \tfrac34\mathbf q^n+\tfrac14E(\mathbf q^{(1)}),\\
   \mathbf q^{n+1} &= \tfrac13\mathbf q^n+\tfrac23E(\mathbf q^{(2)}).
   \end{aligned}

Parenthesized superscripts :math:`(1)` and :math:`(2)` label internal stages,
not powers or completed steps. The unparenthesized :math:`n+1` labels the
next completed time step. Cell indices are suppressed here because the
formulas act on the whole array. The second-stage weights are three quarters
on the old state and one quarter on an Euler update of stage one; the final
weights are one third on the old state and two thirds on an Euler update of
stage two. Each pair is nonnegative and sums to one, making a convex average.

“Strong stability preserving” means that suitable stability properties of a
forward Euler update carry over to these combinations under the appropriate
time-step restriction. It does not mean unconditional stability or an
automatic guarantee for arbitrary dry states. The temporal method is third
order for a sufficiently smooth semidiscrete solution; the first-order
spatial flux still limits the accuracy of the combined solver.

.. code-block:: text

   def rk3_step(q: ℝ[3, m, n], g: ℝ, dx: ℝ, dy: ℝ, Δt: ℝ): ℝ[3, m, n]:
       q1: ℝ[3, m, n] = q + Δt * rhs(q, g, dx, dy)
       q2: ℝ[3, m, n] = 0.75 * q + 0.25 * (q1 + Δt * rhs(q1, g, dx, dy))
       q_next: ℝ[3, m, n] = q / 3.0 + (2.0 / 3.0) * (q2 + Δt * rhs(q2, g, dx, dy))
       return q_next

``q1`` and ``q2`` are the two internal states, and ``q_next`` is the next
completed state. Every call to ``rhs`` recomputes the nonlinear quantities
from its own argument. The method advances x and y fluxes together; there is
no separate directional time step.

Choosing a time step
~~~~~~~~~~~~~~~~~~~~

The Courant–Friedrichs–Lewy (CFL) idea compares how far information can travel
in one step with the size of a cell. In two dimensions both directional
contributions matter. Define a dimensionless speed budget
:math:`\mathcal C(\mathbf q)` by

.. math::

   \mathcal C(\mathbf q)=\Delta t\left(
     \frac{\max_{i,j}(|u_{ij}|+c_{ij})}{\Delta x}
     +\frac{\max_{i,j}(|v_{ij}|+c_{ij})}{\Delta y}\right).

Each maximum ranges over every grid cell of the state being checked. A
conservative target for this small demonstration is

.. math::

   \mathcal C(\mathbf q)\leq 0.2.

This is a chosen safety budget, not a universal sharp stability limit. The
standalone program uses a fixed time step; it does not calculate this budget
at runtime or adapt the step. When changing parameters, check the budget and
positive, finite depth at every RHS input (the old state and both internal
stages), as well as the completed state.

For the uniform reference state at rest, the program's parameters give

.. math::

   N_x=N_y=24, \quad H=g=1, \quad \Delta t=0.003,
   \qquad c=1, \qquad
   \mathcal C=0.003(24+24)=0.144.

This calculation describes the uniform reference state, not a measured
maximum over the bump simulation. The bump increases the local wave speed,
and evolving velocities also affect the budget. Grid refinement, larger
amplitude, larger gravity, or longer steps require a fresh check.

Starting from a smooth periodic bump
------------------------------------

A Gaussian written using ordinary distance from the centre need not join
smoothly across a periodic boundary. The example instead uses cosines inside
an exponential, so the height and its derivatives match at opposite edges.
The initial condition is

.. math::

   B(x,y)=\exp\!\left(
     \frac{\cos(2\pi(x-x_c))+\cos(2\pi(y-y_c))-2}{\beta}
   \right),
   \qquad
   h(x,y,0)=H+A B(x,y), \qquad u(x,y,0)=v(x,y,0)=0,

   x_c=y_c=0.5, \qquad \beta=0.12, \qquad A=0.05, \qquad H=1.

Here :math:`B` is the dimensionless bump shape, :math:`\exp` is the natural
exponential, :math:`\cos` is cosine, and :math:`\pi` is the usual circle
constant (cosine arguments are in radians). The constants :math:`x_c` and
:math:`y_c` set the centre. The positive width parameter :math:`\beta` is the
literal ``0.12`` in the code; smaller values give a narrower bump. The
amplitude :math:`A`, named ``amplitude``, is the maximum surface displacement
of the continuous initial profile. The continuous bump has value one at its
centre. The even-sized cell-centred grid does not sample that centre exactly,
so its sampled maximum need not equal the continuous peak.

``initial_state`` samples the smooth profile at cell centres as an
approximation to the initial cell averages; it does not integrate the profile
over each cell. It sets both momentum arrays to zero, so initially the water
is at rest. The nested for-expressions construct the x and y dependence of
every cell. ``pi`` stores the numerical value of :math:`\pi`, ``bump`` stores
:math:`B`, and ``zero`` is a zero array with the same shape as ``h``.

.. code-block:: text

   def initial_state(Nx: ℕ, Ny: ℕ, H: ℝ, amplitude: ℝ): ℝ[3, Nx, Ny]:
       x: ℝ[Nx] = for i:ℕ(Nx) -> (i + 0.5) / Nx
       y: ℝ[Ny] = for j:ℕ(Ny) -> (j + 0.5) / Ny
       pi: ℝ = 3.141592653589793
       bump: ℝ[Nx, Ny] = for i:ℕ(Nx) -> for j:ℕ(Ny) -> exp(
           (cos(2 * pi * (x[i] - 0.5)) + cos(2 * pi * (y[j] - 0.5)) - 2.0) / 0.12)
       h: ℝ[Nx, Ny] = H + amplitude * bump
       zero: ℝ[Nx, Ny] = 0.0 * h
       return [h, zero, zero]

The positive bump adds water relative to the uniform reference depth. Its
spatial mean is not zero, so the mean initial depth is greater than
:math:`H`. Conservation preserves this initial mean; it does not force the
mean to return to the reference depth.

Running and interpreting the example
------------------------------------

From the repository root, after installing Physika, run

.. code-block:: bash

   physika tutorials/shallow_water_rk3.phyk

To inspect the generated PyTorch without executing the simulation, use

.. code-block:: bash

   physika tutorials/shallow_water_rk3.phyk --print-code

If copying the complete listing below into another directory, save it as
``shallow_water_rk3.phyk`` and pass that path to ``physika`` instead.

The globals ``Nx`` and ``Ny`` set the cell counts, ``H`` the reference depth,
``g`` gravity, ``dx`` and ``dy`` the cell widths, and ``Δt`` the time step.
``steps`` is the number of updates. ``initial`` stores the initial conserved
state; ``final`` starts from that state and is replaced on each iteration.
The loop index ``step`` counts the updates, starting at zero. The final
simulation time :math:`t_{\mathrm{end}}` is

.. math::

   t_{\mathrm{end}}=40\Delta t=0.12.

The only scalar diagnostic printed by this program is
``mean_depth_change``. In mathematical notation, call it
:math:`\delta\overline h`:

.. math::

   \delta\overline h=\frac1{N_xN_y}
   \sum_{i=0}^{N_x-1}\sum_{j=0}^{N_y-1}
   \left(h^{\mathrm{final}}_{ij}-h^{\mathrm{initial}}_{ij}\right).

The superscripts “initial” and “final” identify the depth components of those
two arrays; the sums include every cell. Since all cells have equal area,
this arithmetic mean measures the change in mean depth. It should be near
zero up to floating-point error because of the shared periodic fluxes. A
CPU run with the default single-precision tensors gave approximately
``4.24e-7``; the exact roundoff depends on the execution environment. This
small drift checks conservation, not the accuracy of wave propagation.
The program itself does not produce a plot or a stage-by-stage stability
report; the tests below check the intermediate states.

To interpret the state in your own diagnostics, ``final[0]`` is depth,
``final[0] - H`` is surface displacement, and the two momentum components
divided by ``final[0]`` recover velocities. A smooth surface disturbance
should drive outward flow and gravity waves. Numerical diffusion smooths
those waves, and periodic wrapping matters once waves reach an edge. These
are qualitative expectations, not measured snapshots supplied by this page.

Limits and useful checks
~~~~~~~~~~~~~~~~~~~~~~~~

* The spatial method is deliberately diffusive. Third-order time stepping
  does not make the spatial approximation third order. Resolve the bump on
  progressively finer grids and reduce the time step consistently before
  trusting fine details.
* Division by depth and its square root require positive depth. There is no
  dry-cell treatment, positivity repair, or wetting-and-drying model here.
  Check depth and finite values at internal stages as well as completed steps.
* Monitor the two-dimensional CFL budget when changing the setup. A fixed
  time step that suits one grid or amplitude may fail for another.
* Conservation of mean depth alone cannot establish accuracy. Uniform water
  at rest should remain unchanged; periodic flux cancellation should also
  conserve both total momentum components up to roundoff. Grid and time-step
  comparisons test different errors from these conservation checks.
* This is a flat-bed, depth-averaged gravity-wave demonstration, not a resolved
  three-dimensional simulation, a model of breaking waves, or a reproduction
  of a particular paper's numerical scheme.

Checking the implementation
---------------------------

From the repository root, run the numerical tests with

.. code-block:: bash

   python -m pytest tests/test_shallow_water_rk3.py -q

These tests compile the Physika source and exercise its generated functions.
They check uniform water at rest and in motion, conservation of water and
both momentum components, symmetry when swapping x and y, and pressure-driven
acceleration. They also inspect every Runge–Kutta stage of the default example
for positive finite depth and the CFL budget. In the checked run, the maximum
budget was approximately ``0.1471``, below the chosen ``0.2`` target.

The time-accuracy tests keep the grid fixed while reducing the time step.
One compares a transported shear profile with its exact solution for the
spatially discretized equations; another compares successively refined runs
of the nonlinear system. This isolates time-stepping error from spatial
error. Both checks showed approximately third-order temporal convergence;
neither changes the first-order accuracy of the spatial flux.

On this checkout, the CLI reports that some constructs fall back from CIC
verification to the standard type checker and code generator, then reports
``No type errors found``. The example runs through that supported path; these
numerical tests are not a claim of full formal verification.

Full code
---------

The following is the standalone ``tutorials/shallow_water_rk3.phyk`` program.

.. code-block:: text

   # 2D shallow water: q_t + F(q)_x + G(q)_y = 0 on a periodic unit square.
   # q = [h, hu, hv]: depth and depth-integrated momenta; u, v are velocities.
   # This depth-averaged, flat-bed model omits viscosity and external forcing.
   # It is a reduced model, not a full 3D flow or a reproduction of Silvestri's scheme.
   # All quantities are dimensionless; the reference depth H and gravity g are 1.
   # q[k, i, j] stores component k in cell (i, j). For each component field,
   # axis 0 is x and axis 1 is y; roll(w, -1, axis) selects the next periodic cell.

   # At each positive face, use max(a_left, a_right) = (a + b + |a - b|) / 2.
   def face_speed(a: ℝ[m, n], axis: ℕ): ℝ[m, n]:
       neighbor: ℝ[m, n] = roll(a, -1, axis)
       return 0.5 * (a + neighbor + abs(a - neighbor))

   # For one conserved component w, f and flux_y are its x and y physical fluxes.
   # Rusanov flux: F_face = (f_left + f_right) / 2 - a_face * (w_right - w_left) / 2.
   # Piecewise-constant states give first-order spatial accuracy and numerical diffusion.
   # Shared face fluxes cancel between adjacent cells in the negative divergence.
   def divergence(w: ℝ[m, n], f: ℝ[m, n], flux_y: ℝ[m, n],
                  ax: ℝ[m, n], ay: ℝ[m, n], dx: ℝ, dy: ℝ): ℝ[m, n]:
       F: ℝ[m, n] = 0.5 * (f + roll(f, -1, 0))
       G: ℝ[m, n] = 0.5 * (flux_y + roll(flux_y, -1, 1))
       F = F - 0.5 * ax * (roll(w, -1, 0) - w)
       G = G - 0.5 * ay * (roll(w, -1, 1) - w)
       return -(F - roll(F, 1, 0)) / dx - (G - roll(G, 1, 1)) / dy

   # q_t = L(q), with physical fluxes
   #   F = [hu, hu * u + g * h² / 2, hu * v],
   #   G = [hv, hv * u, hv * v + g * h² / 2].
   # Recompute velocities and gravity-wave speed c from this stage's entire state.
   # The divisions and square root require h > 0; there is no dry-bed treatment.
   def rhs(q: ℝ[3, m, n], g: ℝ, dx: ℝ, dy: ℝ): ℝ[3, m, n]:
       h: ℝ[m, n] = q[0]
       hu: ℝ[m, n] = q[1]
       hv: ℝ[m, n] = q[2]
       u: ℝ[m, n] = hu / h
       v: ℝ[m, n] = hv / h
       c: ℝ[m, n] = sqrt(g * h)
       ax: ℝ[m, n] = face_speed(abs(u) + c, 0)
       ay: ℝ[m, n] = face_speed(abs(v) + c, 1)
       pressure: ℝ[m, n] = 0.5 * g * h * h
       dh: ℝ[m, n] = divergence(h, hu, hv, ax, ay, dx, dy)
       dhu: ℝ[m, n] = divergence(hu, hu * u + pressure, hv * u, ax, ay, dx, dy)
       dhv: ℝ[m, n] = divergence(hv, hu * v, hv * v + pressure, ax, ay, dx, dy)
       return [dh, dhu, dhv]

   # SSPRK(3,3): three coupled stages, third-order in time for the semidiscrete ODE.
   def rk3_step(q: ℝ[3, m, n], g: ℝ, dx: ℝ, dy: ℝ, Δt: ℝ): ℝ[3, m, n]:
       q1: ℝ[3, m, n] = q + Δt * rhs(q, g, dx, dy)
       q2: ℝ[3, m, n] = 0.75 * q + 0.25 * (q1 + Δt * rhs(q1, g, dx, dy))
       q_next: ℝ[3, m, n] = q / 3.0 + (2.0 / 3.0) * (q2 + Δt * rhs(q2, g, dx, dy))
       return q_next

   # Start at rest with a smooth periodic bump, sampled at cell centers.
   def initial_state(Nx: ℕ, Ny: ℕ, H: ℝ, amplitude: ℝ): ℝ[3, Nx, Ny]:
       x: ℝ[Nx] = for i:ℕ(Nx) -> (i + 0.5) / Nx
       y: ℝ[Ny] = for j:ℕ(Ny) -> (j + 0.5) / Ny
       pi: ℝ = 3.141592653589793
       bump: ℝ[Nx, Ny] = for i:ℕ(Nx) -> for j:ℕ(Ny) -> exp(
           (cos(2 * pi * (x[i] - 0.5)) + cos(2 * pi * (y[j] - 0.5)) - 2.0) / 0.12)
       h: ℝ[Nx, Ny] = H + amplitude * bump
       zero: ℝ[Nx, Ny] = 0.0 * h
       return [h, zero, zero]

   Nx: ℕ = 24
   Ny: ℕ = 24
   H: ℝ = 1.0
   g: ℝ = 1.0
   dx: ℝ = 1.0 / Nx
   dy: ℝ = 1.0 / Ny
   # Fixed step for this small bump. The 2D Courant number is
   # Δt * (max(|u| + c) / dx + max(|v| + c) / dy), with maxima over cells.
   # Check it and h > 0 at every stage when changing the setup; neither is enforced here.
   Δt: ℝ = 0.003
   steps: ℕ = 40
   initial: ℝ[3, Nx, Ny] = initial_state(Nx, Ny, H, 0.05)
   final: ℝ[3, Nx, Ny] = initial
   for step:ℕ(steps):
       final = rk3_step(final, g, dx, dy, Δt)

   # Periodic flux cancellation conserves mean depth, up to floating-point roundoff.
   mean_depth_change: ℝ = mean(final[0] - initial[0])
   mean_depth_change
